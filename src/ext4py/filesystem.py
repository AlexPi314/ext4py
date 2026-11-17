"""Чтение inode, каталогов, путей, ссылок и потоков содержимого EXT4.

Файловая система использует общий синхронный поток Image.
Разрешение ссылок не обращается к путям файловой системы хоста.
"""

from collections import deque
from dataclasses import dataclass
import io
import stat
import struct
from typing import Iterator

from .blocks import BlockMap
from .constants import (
    INCOMPAT_64BIT,
    INCOMPAT_FILETYPE,
    IO_CHUNK_SIZE,
    MAX_SYMLINKS,
    MAX_SYMLINK_SIZE,
    ROOT_INODE,
    RO_COMPAT_METADATA_CSUM,
    UNSUPPORTED_INODE_DATA,
)
from .errors import CorruptImageError, Ext4Error, UnsupportedFeatureError
from .image import Image
from .structures import Inode, Superblock, crc32c, u16, u32


@dataclass(frozen=True)
class DirectoryEntry:
    """Имя каталожной записи и номер её inode.

    Attributes:
        name: Имя с surrogateescape для недекодируемых исходных байтов.
        inode: Номер объекта; метаданные загружаются отдельно.
    """

    name: str
    inode: int


class FileSystem:
    """Синхронный интерфейс чтения одной файловой системы EXT4.

    Attributes:
        image: Заимствованное представление байтов раздела.
        sb: Проверенная модель суперблока.

    Объект не закрывает Image самостоятельно. Один поток Image не следует
    одновременно использовать из нескольких потоков исполнения.
    """

    def __init__(self, image: Image):
        """Прочитать суперблок и подготовить ленивое чтение групп.

        Args:
            image: Открытое представление с адресами относительно начала EXT4.

        Ошибки формата суперблока передаются вызывающей стороне.
        """
        self.image = image
        self.sb = Superblock.parse(image.read_at(1024, 1024), image.size)
        self._tables: dict[int, int] = {}

    def read_block(self, number: int) -> bytes:
        """Прочитать один физический блок после проверки его номера.

        Args:
            number: Положительный номер блока внутри объявленной длины EXT4.

        Returns:
            Полный блок длиной sb.block_size байт.

        Raises:
            CorruptImageError: Номер вне EXT4 или образ усечён.
        """
        if number <= 0 or number >= self.sb.blocks_count:
            raise CorruptImageError(f"Блок {number} вне границ файловой системы")
        return self.image.read_at(number * self.sb.block_size, self.sb.block_size)

    def _inode_table(self, group: int) -> int:
        """Найти начало таблицы inode группы и закешировать адрес.

        Args:
            group: Номер группы от нуля до sb.groups_count - 1.

        Returns:
            Номер физического блока начала таблицы, включая старшую часть 64bit.

        Дескриптор и диапазон таблицы проверяются до записи адреса в кеш.
        """
        if group in self._tables:
            return self._tables[group]
        sb = self.sb
        if not 0 <= group < sb.groups_count:
            raise CorruptImageError("Номер группы вне границ файловой системы")
        offset = (sb.first_data_block + 1) * sb.block_size + group * sb.descriptor_size
        if offset + sb.descriptor_size > sb.blocks_count * sb.block_size:
            raise CorruptImageError("Дескриптор группы вне границ EXT4")
        data = self.image.read_at(offset, sb.descriptor_size)
        if sb.ro_compat & RO_COMPAT_METADATA_CSUM:
            cleared = data[:0x1E] + b"\0\0" + data[0x20:]
            checksum = crc32c(cleared, crc32c(struct.pack("<I", group), sb.checksum_seed)) & 0xFFFF
            if checksum != u16(data, 0x1E):
                raise CorruptImageError(f"Не совпадает контрольная сумма дескриптора группы {group}")
        # При flex_bg таблица может находиться вне собственной группы.
        table = u32(data, 8)
        if sb.incompat & INCOMPAT_64BIT:
            table |= u32(data, 0x28) << 32
        table_blocks = (sb.inodes_per_group * sb.inode_size + sb.block_size - 1) // sb.block_size
        if table <= 0 or table + table_blocks > sb.blocks_count:
            raise CorruptImageError("Таблица inode выходит за границы EXT4")
        self._tables[group] = table
        return self._tables[group]

    def inode(self, number: int) -> Inode:
        """Загрузить и проверить индексный узел по номеру.

        Args:
            number: Номер inode, начиная с 1.

        Returns:
            Inode, на который допустимо переходить из каталога.

        Raises:
            CorruptImageError: Номер, CRC или запись узла некорректны.

        Битовая карта выделения отдельно не проверяется.
        """
        sb = self.sb
        if not 1 <= number <= sb.inodes_count:
            raise CorruptImageError(f"Недопустимый номер inode: {number}")
        group, index = divmod(number - 1, sb.inodes_per_group)
        data = self.image.read_at(self._inode_table(group) * sb.block_size + index * sb.inode_size,
                                  sb.inode_size)
        if sb.ro_compat & RO_COMPAT_METADATA_CSUM:
            stored = u16(data, 0x7C)
            # Сохранённые поля CRC обнуляются перед повторным вычислением.
            cleared = bytearray(data)
            cleared[0x7C:0x7E] = b"\0\0"
            has_high = len(data) > 128 and u16(data, 0x80) >= 4
            if has_high:
                stored |= u16(data, 0x82) << 16
                cleared[0x82:0x84] = b"\0\0"
            seed = crc32c(struct.pack("<II", number, u32(data, 0x64)), sb.checksum_seed)
            calculated = crc32c(cleared, seed)
            if stored != (calculated if has_high else calculated & 0xFFFF):
                raise CorruptImageError(f"Не совпадает контрольная сумма inode {number}")
        inode = Inode.parse(number, data)
        if not stat.S_IFMT(inode.mode) or not inode.links:
            raise CorruptImageError(f"Каталог ссылается на неиспользуемый inode {number}")
        return inode

    @staticmethod
    def _check_data(inode: Inode) -> None:
        """Отклонить не реализованный способ хранения данных inode.

        Args:
            inode: Узел перед чтением его содержимого или цели ссылки.

        Raises:
            UnsupportedFeatureError: Обнаружены неподдерживаемые флаги данных или имён.
        """
        unsupported = inode.flags & UNSUPPORTED_INODE_DATA
        if unsupported:
            raise UnsupportedFeatureError(f"Неподдерживаемые флаги inode {inode.number}: 0x{unsupported:x}")

    def iter_data(self, inode: Inode, chunk_size: int = IO_CHUNK_SIZE) -> Iterator[bytes]:
        """Выдать логическое содержимое объекта ограниченными порциями.

        Args:
            inode: Обычный файл, каталог или блочная символьная ссылка.
            chunk_size: Положительный максимальный размер порции в байтах.

        Yields:
            Байты до inode.size; пропуски и unwritten заменены нулями.

        Raises:
            ValueError: Размер порции не положителен.
            Ext4Error: Тип или способ хранения не допускает чтения.
            CorruptImageError: Нарушена карта блоков или доступность данных.
        """
        if chunk_size <= 0:
            raise ValueError("chunk_size должен быть положительным")
        self._check_data(inode)
        if not (stat.S_ISREG(inode.mode) or stat.S_ISDIR(inode.mode) or stat.S_ISLNK(inode.mode)):
            raise Ext4Error(f"Нельзя читать содержимое объекта типа «{inode.kind}»")
        block_size = self.sb.block_size
        cursor = 0
        for logical, count, physical in BlockMap(self.sb, self.read_block, inode).ranges():
            start = logical * block_size
            if start < cursor:
                raise CorruptImageError("Перекрытие диапазонов файла")
            if cursor >= inode.size:
                return
            gap_end = min(start, inode.size)
            while cursor < gap_end:
                length = min(chunk_size, gap_end - cursor)
                yield bytes(length)
                cursor += length
            end = min((logical + count) * block_size, inode.size)
            while cursor < end:
                length = min(chunk_size, end - cursor)
                if physical is None:
                    yield bytes(length)
                else:
                    yield self.image.read_at(physical * block_size + cursor - start, length)
                cursor += length
        while cursor < inode.size:
            length = min(chunk_size, inode.size - cursor)
            yield bytes(length)
            cursor += length

    def directory(self, inode: Inode) -> Iterator[DirectoryEntry]:
        """Разобрать все блоки каталога совместимым линейным обходом.

        Args:
            inode: Узел каталога с размером, кратным размеру блока.

        Yields:
            DirectoryEntry для занятых записей, включая скрытые имена и точки.

        Raises:
            Ext4Error: Передан не каталог.
            CorruptImageError: Некорректны длины, имя, адрес или CRC листа.
        """
        if not stat.S_ISDIR(inode.mode):
            raise Ext4Error(f"Inode {inode.number}: это не каталог")
        if inode.size % self.sb.block_size:
            raise CorruptImageError("Размер каталога не кратен размеру блока")
        # Внутренние HTree-узлы выглядят как служебные записи; читаются все листья.
        for data in self.iter_data(inode, self.sb.block_size):
            if self.sb.ro_compat & RO_COMPAT_METADATA_CSUM and data[-12:-4] == b"\0\0\0\0\x0c\0\0\xde":
                seed = crc32c(struct.pack("<II", inode.number, inode.generation), self.sb.checksum_seed)
                if crc32c(data[:-12], seed) != u32(data, len(data) - 4):
                    raise CorruptImageError("Не совпадает контрольная сумма блока каталога")
            pos = 0
            while pos < len(data):
                if pos + 8 > len(data):
                    raise CorruptImageError("Усечён заголовок записи каталога")
                number, record_len = struct.unpack_from("<IH", data, pos)
                # В 16-битном rec_len полный блок 64 КиБ кодируется особым образом.
                if self.sb.block_size == 65536:
                    if record_len in (0, 65535):
                        record_len = 65536
                    else:
                        record_len = (record_len & 65532) | ((record_len & 3) << 16)
                name_len = data[pos + 6] if self.sb.incompat & INCOMPAT_FILETYPE else u16(data, pos + 6)
                if record_len < 8 or record_len % 4 or pos + record_len > len(data):
                    raise CorruptImageError("Некорректная длина записи каталога")
                if number:
                    if not name_len or name_len > 255 or name_len > record_len - 8:
                        raise CorruptImageError("Некорректная длина имени в каталоге")
                    raw_name = data[pos + 8:pos + 8 + name_len]
                    if b"/" in raw_name or b"\0" in raw_name or number > self.sb.inodes_count:
                        raise CorruptImageError("Некорректное имя или номер inode в каталоге")
                    yield DirectoryEntry(raw_name.decode("utf-8", "surrogateescape"), number)
                pos += record_len

    def readlink(self, inode: Inode) -> str:
        """Прочитать цель символьной ссылки без её разыменования.

        Args:
            inode: Узел короткой или блочной символьной ссылки.

        Returns:
            Текст цели с сохранением недекодируемых байтов через surrogateescape.

        Raises:
            Ext4Error: Узел не является ссылкой или способ хранения не поддержан.
            CorruptImageError: Неверны длина, содержимое или карта ссылки.
        """
        if not stat.S_ISLNK(inode.mode):
            raise Ext4Error("Указанный объект не является символьной ссылкой")
        self._check_data(inode)
        if inode.size > MAX_SYMLINK_SIZE or not inode.size:
            raise CorruptImageError("Некорректная длина символьной ссылки")
        # Блок внешних атрибутов не является блоком содержимого короткой ссылки.
        ea_blocks = self.sb.block_size // 512 if inode.file_acl else 0
        if inode.blocks == ea_blocks:
            if inode.size > 60:
                raise CorruptImageError("Короткая ссылка не помещается в inode")
            data = inode.block_data[:inode.size]
        else:
            data = b"".join(self.iter_data(inode))
        if b"\0" in data:
            raise CorruptImageError("Нулевой байт в символьной ссылке")
        return data.decode("utf-8", "surrogateescape")

    def resolve(self, path: str, follow_final: bool = True) -> Inode:
        """Разрешить абсолютный путь только внутри образа.

        Args:
            path: Путь с разделителем '/', без нулевых байтов.
            follow_final: Разыменовывать последнюю ссылку; завершающий '/'
                в любом случае требует каталога и перехода по ссылке.

        Returns:
            Inode найденного объекта.

        Raises:
            Ext4Error: Путь недопустим, отсутствует или превышен предел ссылок.
            CorruptImageError: Найдены повреждённые структуры или повтор имени.

        Относительная цель ссылки отсчитывается от её родителя, абсолютная —
        от корня EXT4. Компонент '..' обрабатывается после предыдущих ссылок.
        """
        if not path.startswith("/") or "\0" in path:
            raise Ext4Error("Путь внутри образа должен быть абсолютным и не содержать нулевых байтов")
        pending = deque(part for part in path.split("/") if part)
        current = self.inode(ROOT_INODE)
        links = 0
        require_directory = path.endswith("/")
        while pending:
            component = pending.popleft()
            if not stat.S_ISDIR(current.mode):
                raise Ext4Error(f"Компонент пути не является каталогом: {component!r}")
            if component == ".":
                continue
            if component == ".." and current.number == ROOT_INODE:
                continue
            matches = (entry for entry in self.directory(current) if entry.name == component)
            entry = next(matches, None)
            if entry is None:
                raise Ext4Error(f"Объект не найден: {path!r}")
            if next(matches, None) is not None:
                raise CorruptImageError("Повторяющееся имя в каталоге")
            child = self.inode(entry.inode)
            if stat.S_ISLNK(child.mode) and (pending or follow_final or require_directory):
                links += 1
                if links > MAX_SYMLINKS:
                    raise Ext4Error(f"Превышено число переходов по символьным ссылкам ({MAX_SYMLINKS})")
                target = self.readlink(child)
                if target.endswith("/") and not pending:
                    require_directory = True
                # extendleft разворачивает порядок, поэтому цель заранее переворачивается.
                pending.extendleft(reversed([part for part in target.split("/") if part]))
                if target.startswith("/"):
                    current = self.inode(ROOT_INODE)
            else:
                current = child
        if require_directory and not stat.S_ISDIR(current.mode):
            raise Ext4Error("Путь с завершающим '/' должен указывать на каталог")
        return current

    def open_file(self, inode: Inode) -> io.BufferedReader:
        """Создать буферизованный поток обычного файла.

        Args:
            inode: Узел обычного файла; другие типы отклоняются.

        Returns:
            BufferedReader, который вызывающий код должен закрыть.

        Проверки блоков происходят при чтении возвращённого потока.
        """
        if not stat.S_ISREG(inode.mode):
            raise Ext4Error(f"Ожидался обычный файл, получен объект типа «{inode.kind}»")
        return io.BufferedReader(InodeReader(self.iter_data(inode)))


class InodeReader(io.RawIOBase):
    """Адаптер итератора порций bytes к интерфейсу io.RawIOBase.

    Остаток порции хранится как memoryview; пустые порции пропускаются.
    Завершение итератора означает конец потока.
    """

    def __init__(self, chunks: Iterator[bytes]):
        """Сохранить ленивый источник двоичных порций.

        Args:
            chunks: Итератор bytes; исключения источника передаются читателю.
        """
        super().__init__()
        self.chunks = iter(chunks)
        self.pending = memoryview(b"")

    def readable(self) -> bool:
        """Сообщить потоковым обёрткам о поддержке чтения."""
        return True

    def readinto(self, buffer: bytearray) -> int:
        """Заполнить изменяемый буфер, сохранив остаток текущей порции.

        Args:
            buffer: Двоичный буфер, доступный для записи срезами.

        Returns:
            Число записанных байт; ноль при EOF или пустом буфере.

        Raises:
            ValueError: Поток закрыт.

        Исключение источника не скрывается, даже после чтения части порции.
        """
        if self.closed:
            raise ValueError("I/O operation on closed file")
        written = 0
        while written < len(buffer):
            while not self.pending:
                chunk = next(self.chunks, None)
                if chunk is None:
                    return written
                self.pending = memoryview(chunk)
            count = min(len(buffer) - written, len(self.pending))
            buffer[written:written + count] = self.pending[:count]
            self.pending = self.pending[count:]
            written += count
        return written
