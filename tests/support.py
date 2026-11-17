"""Синтетические метаданные и небольшие raw-образы для модульных проверок.

Фикстуры используют блоки 1 КиБ и заранее известную раскладку.
Образы независимого интеграционного стенда создаются через mke2fs.
"""

from pathlib import Path
import struct
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

from ext4py.structures import Inode, Superblock


def super_bytes() -> bytes:
    """Вернуть минимальный суперблок 1-КиБ EXT4 без контрольных сумм.

    Геометрия задаёт 128 блоков и 16 inode; остальные поля пригодны
    для изолированных проверок декодирования, а не для монтирования.
    """
    data = bytearray(1024)
    for offset, value in {0: 16, 4: 128, 0x14: 1, 0x20: 128, 0x28: 16,
                          0x4C: 1, 0x60: 0x42}.items():
        struct.pack_into('<I', data, offset, value)
    for offset, value in {0x38: 0xEF53, 0x3A: 1, 0x58: 128}.items():
        struct.pack_into('<H', data, offset, value)
    return bytes(data)


def superblock() -> Superblock:
    """Декодировать super_bytes и вернуть модель геометрии 128-КиБ образа."""
    return Superblock.parse(super_bytes(), 128 * 1024)


def inode(block_data: bytes, size: int = 1024, flags: int = 0x80000) -> Inode:
    """Создать модель обычного файла для проверок карты блоков.

    Args:
        block_data: Поле i_block, дополняемое нулями до 60 байт.
        size: Логический размер файла в байтах.
        flags: Флаги узла; по умолчанию карта extent.

    Returns:
        Inode с номером 12 и известными базовыми метаданными.
    """
    return Inode(12, 0o100644, 1000, 1000, size, 0, 1, 2, flags,
                 block_data.ljust(60, b'\0'), 0, 0)


def extent_node(entries: list[tuple[int, int, int]], depth: int = 0, size: int = 60) -> bytes:
    """Сформировать байты листа или индекса extent-дерева.

    Args:
        entries: Для листа тройки (logical, length, physical), для индекса
            (logical, child, 0). Значения можно намеренно сделать неверными.
        depth: Глубина узла; ноль обозначает лист.
        size: Длина корня или внешнего блока в байтах.

    Returns:
        Буфер узла без автоматически рассчитанной CRC.
    """
    data = bytearray(size)
    struct.pack_into('<HHHHI', data, 0, 0xF30A, len(entries), (size - 12) // 12, depth, 0)
    for index, (logical, length, physical) in enumerate(entries):
        if depth:
            struct.pack_into('<IIHH', data, 12 + index * 12, logical, length, 0, 0)
        else:
            struct.pack_into('<IHHI', data, 12 + index * 12, logical, length, physical >> 32, physical & 0xFFFFFFFF)
    return bytes(data)


def directory_block(entries: list[tuple[int, bytes]], block_size: int = 1024, filetype: bool = True) -> bytes:
    """Сформировать блок старых или новых записей каталога.

    Args:
        entries: Пары номера inode и исходных байтов имени.
        block_size: Длина результата; допускается блок 64 КиБ.
        filetype: Использовать 8-битную длину имени вместо старой 16-битной.

    Returns:
        Блок с выровненными записями и последней записью до конца блока.
    """
    data = bytearray(block_size)
    position = 0
    for index, (number, name) in enumerate(entries):
        length = (8 + len(name) + 3) & ~3
        if index == len(entries) - 1:
            length = block_size - position
        encoded = 65535 if length == 65536 else length
        struct.pack_into('<IH', data, position, number, encoded)
        if filetype:
            data[position + 6] = len(name)
        else:
            struct.pack_into('<H', data, position + 6, len(name))
        data[position + 8:position + 8 + len(name)] = name
        position += length
    return bytes(data)


class RawFixture:

    """Небольшой изменяемый raw-образ для независимых модульных проверок.

    Таблица inode находится в блоке 5, данные начинаются с блока 20.
    Поддерживаются 32 inode и блоки 1 КиБ; это фиксированная тестовая раскладка.
    """

    def __init__(self):
        """Подготовить 128-КиБ буфер с суперблоком и дескриптором группы."""
        self.data = bytearray(128 * 1024)
        self.data[1024:2048] = super_bytes()
        struct.pack_into('<I', self.data, 1024, 32)
        struct.pack_into('<I', self.data, 1024 + 0x28, 32)
        struct.pack_into('<I', self.data, 2048 + 8, 5)
        self.next_block = 20

    def put(self, number: int, mode: int, content: bytes = b'', links: int = 1, fast: bool = False):
        """Записать inode и содержимое в синтетический образ.

        Args:
            number: Номер записи в тестовой таблице.
            mode: Тип и права объекта в битах UNIX.
            content: Байты данных либо цель короткой ссылки.
            links: Число жёстких ссылок, в том числе ноль для повреждения.
            fast: Хранить цель ссылки прямо в i_block без блоков данных.

        Returns:
            Текущая фикстура для последовательного наполнения через цепочку вызовов.

        Ёмкость области данных и корректность аргументов задаёт сам тест.
        """
        offset = 5 * 1024 + (number - 1) * 128
        node = bytearray(128)
        struct.pack_into('<H', node, 0, mode)
        struct.pack_into('<I', node, 4, len(content))
        struct.pack_into('<H', node, 0x1A, links)
        if fast:
            node[0x28:0x28 + len(content)] = content
        else:
            blocks = (len(content) + 1023) // 1024
            struct.pack_into('<I', node, 0x1C, blocks * 2)
            struct.pack_into('<I', node, 0x20, 0x80000)
            node[0x28:0x64] = extent_node([(0, blocks, self.next_block)] if blocks else [])
            self.data[self.next_block * 1024:self.next_block * 1024 + len(content)] = content
            self.next_block += blocks
        self.data[offset:offset + 128] = node
        return self

    def filesystem(self):
        """Вернуть FileSystem, читающий копию текущего буфера через настоящий Image."""
        import io
        from ext4py.filesystem import FileSystem
        from ext4py.image import Image
        return FileSystem(Image(io.BytesIO(self.data)))


def path_fixture():
    """Создать дерево с каталогом, файлом и относительными/абсолютными ссылками.

    Returns:
        RawFixture с известными номерами inode для проверки поиска и экспорта.
    """
    fixture = RawFixture()
    fixture.put(2, 0o40755, directory_block([(2, b'.'), (2, b'..'), (3, b'dir'),
                                          (12, b'file'), (13, b'rel'), (14, b'abs'),
                                          (15, b'dirlink'), (16, b'rootlink')]))
    fixture.put(3, 0o40755, directory_block([(3, b'.'), (2, b'..'), (12, b'child')]))
    fixture.put(12, 0o100640, b'expected-data')
    fixture.put(13, 0o120777, b'dir/child', fast=True)
    fixture.put(14, 0o120777, b'/dir/child', fast=True)
    fixture.put(15, 0o120777, b'dir/', fast=True)
    fixture.put(16, 0o120777, b'/', fast=True)
    return fixture
