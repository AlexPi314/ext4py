"""Декодирование little-endian метаданных EXT4 и вычисление CRC32C.

Модели содержат необходимые для чтения поля. Проверка структуры
не заменяет полную проверку распределения блоков файловой системы.
"""

from dataclasses import dataclass
import stat
import struct
import uuid

from .constants import (
    EXT4_MAGIC,
    INCOMPAT_64BIT,
    INCOMPAT_CSUM_SEED,
    INCOMPAT_RECOVER,
    RO_COMPAT_BIGALLOC,
    RO_COMPAT_METADATA_CSUM,
    RO_COMPAT_ORPHAN_PRESENT,
    SUPPORTED_INCOMPAT,
)
from .errors import CorruptImageError, UnsupportedFeatureError


def u16(data: bytes, offset: int) -> int:
    """Прочитать беззнаковое 16-битное число little-endian.

    Args:
        data: Декодируемый буфер байтов.
        offset: Смещение первого байта числа внутри data.

    Returns:
        Целое число без изменения исходного буфера.

    Вызывающий код должен предварительно проверить размер структуры;
    слишком короткий буфер приводит к struct.error.
    """
    return struct.unpack_from("<H", data, offset)[0]


def u32(data: bytes, offset: int) -> int:
    """Прочитать беззнаковое 32-битное число little-endian.

    Args:
        data: Декодируемый буфер байтов.
        offset: Смещение первого байта числа внутри data.

    Returns:
        Целое число без изменения исходного буфера.

    Вызывающий код должен предварительно проверить размер структуры;
    слишком короткий буфер приводит к struct.error.
    """
    return struct.unpack_from("<I", data, offset)[0]


def crc32c(data: bytes, seed: int = 0xFFFFFFFF) -> int:
    """Вычислить CRC32C Castagnoli в соглашении EXT4.

    Args:
        data: Байты очередной части проверяемой структуры.
        seed: Начальное состояние CRC или результат предыдущей части.

    Returns:
        32-битное состояние без финального XOR.

    Функция проверяет ошибки передачи/хранения, а не подлинность данных.
    """
    crc = seed
    for byte in data:
        crc ^= byte
        for _ in range(8):
            crc = (crc >> 1) ^ (0x82F63B78 if crc & 1 else 0)
    return crc


@dataclass(frozen=True)
class Superblock:
    """Неизменяемая геометрия и признаки возможностей EXT4.

    Attributes:
        inodes_count, blocks_count: Общее число inode и блоков.
        block_size: Размер блока в байтах.
        first_data_block: Первый блок данных из суперблока.
        blocks_per_group, inodes_per_group: Ёмкость группы.
        inode_size, descriptor_size: Размеры записей в байтах.
        compat, incompat, ro_compat: Три независимых набора флагов.
        state: Проверенное состояние файловой системы.
        uuid, label: Идентификатор и метка тома.
        checksum_seed: Основа CRC метаданных с учётом csum_seed.
    """

    inodes_count: int
    blocks_count: int
    block_size: int
    first_data_block: int
    blocks_per_group: int
    inodes_per_group: int
    inode_size: int
    descriptor_size: int
    compat: int
    incompat: int
    ro_compat: int
    state: int
    uuid: str
    label: str
    checksum_seed: int

    @classmethod
    def parse(cls, data: bytes, image_size: int) -> "Superblock":
        """Декодировать и проверить основной суперблок.

        Args:
            data: Ровно 1024 байта суперблока без префикса раздела.
            image_size: Доступный размер образа от начала EXT4, в байтах.

        Returns:
            Superblock с геометрией, допустимой для реализованного читателя.

        Raises:
            CorruptImageError: Неверная сигнатура, геометрия, CRC или состояние.
            UnsupportedFeatureError: Обнаружен неподдерживаемый формат.
        """
        data = bytes(data)
        if len(data) != 1024 or u16(data, 0x38) != EXT4_MAGIC:
            raise CorruptImageError("Сигнатура EXT4 не найдена; проверьте raw-формат и --offset")
        log_size = u32(data, 0x18)
        if log_size > 6:
            raise CorruptImageError("Недопустимый размер блока")
        block_size = 1024 << log_size
        compat, incompat, ro = struct.unpack_from("<III", data, 0x5C)
        unknown = incompat & ~SUPPORTED_INCOMPAT
        if unknown:
            raise UnsupportedFeatureError(
                f"Неподдерживаемые INCOMPAT-флаги: 0x{unknown:x} "
                "(например, meta_bg, inline_data, encrypt, casefold)"
            )
        if ro & RO_COMPAT_BIGALLOC:
            raise UnsupportedFeatureError("Файловая система bigalloc не поддерживается")
        if u32(data, 0x48) != 0:
            raise UnsupportedFeatureError("Поддерживаются индексные узлы формата Linux")
        if u32(data, 0x4C) > 1:
            raise UnsupportedFeatureError("Неизвестная ревизия файловой системы")
        state = u16(data, 0x3A)
        if state != 1 or incompat & INCOMPAT_RECOVER or ro & RO_COMPAT_ORPHAN_PRESENT:
            raise CorruptImageError(
                "Файловая система не завершена корректно; журнал не воспроизводится. "
                "Для чтения нужен согласованный образ"
            )
        if ro & RO_COMPAT_METADATA_CSUM:
            if data[0x175] != 1:
                raise UnsupportedFeatureError("Неизвестный алгоритм контрольной суммы EXT4")
            # Поле s_checksum не входит в собственную контрольную сумму.
            if crc32c(data[:0x3FC]) != u32(data, 0x3FC):
                raise CorruptImageError("Не совпадает контрольная сумма суперблока")
        blocks = u32(data, 4) | ((u32(data, 0x150) << 32) if incompat & INCOMPAT_64BIT else 0)
        inodes_count = u32(data, 0)
        first_data_block = u32(data, 0x14)
        blocks_per_group = u32(data, 0x20)
        inodes_per_group = u32(data, 0x28)
        inode_size = u16(data, 0x58) if u32(data, 0x4C) else 128
        desc_size = u16(data, 0xFE) if incompat & INCOMPAT_64BIT else 32
        valid_counts = blocks > 0 and inodes_count > 0
        valid_groups = (
            first_data_block == (1 if block_size == 1024 else 0)
            and 0 < blocks_per_group <= block_size * 8
            and 0 < inodes_per_group <= block_size * 8
        )
        valid_inode_size = 128 <= inode_size <= block_size and inode_size & (inode_size - 1) == 0
        minimum_descriptor_size = 64 if incompat & INCOMPAT_64BIT else 32
        valid_descriptor_size = (
            minimum_descriptor_size <= desc_size <= block_size
            and desc_size & (desc_size - 1) == 0
        )
        if not all((valid_counts, valid_groups, valid_inode_size, valid_descriptor_size)):
            raise CorruptImageError("Некорректная геометрия файловой системы")
        if blocks * block_size > image_size:
            raise CorruptImageError("Образ усечён: размер меньше объявленного в суперблоке")
        groups = (blocks - first_data_block + blocks_per_group - 1) // blocks_per_group
        if not (groups - 1) * inodes_per_group < inodes_count <= groups * inodes_per_group:
            raise CorruptImageError("Количество индексных узлов не соответствует числу групп")
        seed = u32(data, 0x270) if incompat & INCOMPAT_CSUM_SEED else crc32c(data[0x68:0x78])
        return cls(
            inodes_count=inodes_count,
            blocks_count=blocks,
            block_size=block_size,
            first_data_block=first_data_block,
            blocks_per_group=blocks_per_group,
            inodes_per_group=inodes_per_group,
            inode_size=inode_size,
            descriptor_size=desc_size,
            compat=compat,
            incompat=incompat,
            ro_compat=ro,
            state=state,
            uuid=str(uuid.UUID(bytes=data[0x68:0x78])),
            label=data[0x78:0x88].split(b"\0", 1)[0].decode("utf-8", "backslashreplace"),
            checksum_seed=seed,
        )

    @property
    def groups_count(self) -> int:
        """Вернуть число групп с учётом неполной последней группы."""
        return (self.blocks_count - self.first_data_block + self.blocks_per_group - 1) // self.blocks_per_group


@dataclass(frozen=True)
class Inode:
    """Неизменяемые метаданные объекта и его поле i_block.

    Attributes:
        number: Номер inode, начиная с 1.
        mode: Тип и права доступа в формате UNIX.
        uid, gid: Числовые идентификаторы владельца и группы.
        size: Логическая длина объекта в байтах.
        mtime_ns: Время изменения в наносекундах от эпохи Unix.
        links: Число жёстких ссылок.
        blocks: Счётчик i_blocks, объединённый из младшей и старшей частей.
        flags: Признаки inode, задающие способ хранения.
        block_data: 60 байт карты блоков или короткой ссылки.
        file_acl: Адрес внешнего блока атрибутов, если он есть.
        generation: Поколение узла, участвующее в расчёте CRC.
    """

    number: int
    mode: int
    uid: int
    gid: int
    size: int
    mtime_ns: int
    links: int
    blocks: int
    flags: int
    block_data: bytes
    file_acl: int
    generation: int

    @classmethod
    def parse(cls, number: int, data: bytes) -> "Inode":
        """Декодировать запись inode формата Linux.

        Args:
            number: Номер записи, полученный из проверенного диапазона.
            data: Полная запись длиной не менее 128 байт.

        Returns:
            Inode с объединёнными старшими полями и точным временем mtime.

        Raises:
            CorruptImageError: Неверны i_extra_isize или наносекунды.

        CRC и границы номера проверяет FileSystem до вызова этого метода.
        """
        mode = u16(data, 0)
        size = u32(data, 4)
        if stat.S_ISREG(mode) or stat.S_ISDIR(mode):
            size |= u32(data, 0x6C) << 32
        extra_size = u16(data, 0x80) if len(data) > 128 else 0
        if extra_size and (extra_size < 4 or 128 + extra_size > len(data)):
            raise CorruptImageError(f"Некорректный i_extra_isize в inode {number}")
        # Младшие секунды знаковые; дополнительные биты расширяют эпоху.
        seconds = struct.unpack_from("<i", data, 0x10)[0]
        nano = 0
        if extra_size >= 12:
            extra = u32(data, 0x88)
            seconds += (extra & 3) << 32
            nano = extra >> 2
            if nano >= 1_000_000_000:
                raise CorruptImageError(f"Некорректная дата в inode {number}")
        return cls(
            number=number,
            mode=mode,
            uid=u16(data, 2) | (u16(data, 0x78) << 16),
            gid=u16(data, 0x18) | (u16(data, 0x7A) << 16),
            size=size,
            mtime_ns=seconds * 1_000_000_000 + nano,
            links=u16(data, 0x1A),
            blocks=u32(data, 0x1C) | (u16(data, 0x74) << 32),
            flags=u32(data, 0x20),
            block_data=data[0x28:0x64],
            file_acl=u32(data, 0x68) | (u16(data, 0x76) << 32),
            generation=u32(data, 0x64),
        )

    @property
    def kind(self) -> str:
        """Вернуть русское название типа объекта по битам i_mode."""
        return {stat.S_IFREG: "файл", stat.S_IFDIR: "каталог", stat.S_IFLNK: "ссылка",
                stat.S_IFCHR: "символьное устройство", stat.S_IFBLK: "блочное устройство",
                stat.S_IFIFO: "FIFO", stat.S_IFSOCK: "сокет"}.get(stat.S_IFMT(self.mode), "неизвестный")
