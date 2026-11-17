"""Граничные значения геометрии, признаков формата и полей inode."""

import stat
import struct
import unittest

from support import super_bytes
from ext4py.errors import CorruptImageError, UnsupportedFeatureError
from ext4py.structures import Inode, Superblock, crc32c, u16, u32


def super_with(*changes):
    """Изменить выбранные поля базового суперблока.

    Args:
        changes: Тройки (байтовое смещение, формат struct, новое значение).

    Returns:
        Изменяемый буфер для одного граничного или отрицательного сценария.
    """
    data = bytearray(super_bytes())
    for offset, kind, value in changes:
        struct.pack_into(kind, data, offset, value)
    return data


class MetadataTests(unittest.TestCase):

    """Декодирование метаданных и отказ при несовместимой геометрии."""

    def test_binary_readers_use_little_endian(self):
        """Явно декодировать младшие байты чисел первыми."""
        self.assertEqual(u16(b'x\x34\x12', 1), 0x1234)
        self.assertEqual(u32(b'x\x78\x56\x34\x12', 1), 0x12345678)

    def test_crc_empty_preserves_seed(self):
        """Сохранить состояние CRC при пустой очередной части."""
        self.assertEqual(crc32c(b''), 0xFFFFFFFF)
        self.assertEqual(crc32c(b'', 123), 123)

    def test_crc_incremental(self):
        """Согласовать вычисление CRC по целому буферу и частям."""
        data = bytes(range(256))
        self.assertEqual(crc32c(data), crc32c(data[71:], crc32c(data[:71])))

    def test_invalid_superblock_signature(self):
        """Отклонить неверную сигнатуру и неправильную длину суперблока."""
        for data in (bytes(1024), super_bytes()[:-1], super_bytes() + b'x'):
            with self.subTest(size=len(data)), self.assertRaises(CorruptImageError):
                Superblock.parse(data, 131072)

    def test_invalid_block_size_exponent(self):
        """Отклонить размер блока больше поддерживаемого."""
        with self.assertRaisesRegex(CorruptImageError, 'размер блока'):
            Superblock.parse(super_with((0x18, '<I', 7)), 1 << 40)

    def test_bigalloc_refused(self):
        """Отклонить неподдерживаемое кластерное выделение bigalloc."""
        with self.assertRaisesRegex(UnsupportedFeatureError, 'bigalloc'):
            Superblock.parse(super_with((0x64, '<I', 0x200)), 131072)

    def test_non_linux_inode_format_refused(self):
        """Не интерпретировать иной формат inode как Linux."""
        with self.assertRaisesRegex(UnsupportedFeatureError, 'Linux'):
            Superblock.parse(super_with((0x48, '<I', 1)), 131072)

    def test_unknown_revision_refused(self):
        """Отклонить неизвестную ревизию суперблока."""
        with self.assertRaisesRegex(UnsupportedFeatureError, 'ревизия'):
            Superblock.parse(super_with((0x4C, '<I', 2)), 131072)

    def test_original_revision_uses_128_byte_inodes(self):
        """Применить фиксированный размер inode прежней ревизии."""
        value = Superblock.parse(super_with((0x4C, '<I', 0), (0x58, '<H', 0)), 131072)
        self.assertEqual(value.inode_size, 128)
        self.assertEqual(value.descriptor_size, 32)

    def test_unclean_state_bits(self):
        """Отклонить состояния, отличные от согласованного завершения."""
        for state in (0, 2, 3, 4, 5):
            with self.subTest(state=state), self.assertRaises(CorruptImageError):
                Superblock.parse(super_with((0x3A, '<H', state)), 131072)

    def test_orphan_present_refused(self):
        """Не читать том с ожидающей обработки orphan-структурой."""
        with self.assertRaisesRegex(CorruptImageError, 'согласованный'):
            Superblock.parse(super_with((0x64, '<I', 0x10000)), 131072)

    def test_unknown_checksum_algorithm(self):
        """Отклонить неизвестный алгоритм контрольной суммы."""
        data = super_with((0x64, '<I', 0x400))
        data[0x175] = 2
        with self.assertRaisesRegex(UnsupportedFeatureError, 'алгоритм'):
            Superblock.parse(data, 131072)

    def test_invalid_geometry_fields(self):
        """Проверить ограничения количеств и размеров структур."""
        changes = [(0x14, '<I', 0), (0x20, '<I', 8193), (0x28, '<I', 8193),
                   (0x58, '<H', 64), (0x58, '<H', 192), (0x58, '<H', 2048),
                   (0, '<I', 17)]
        for change in changes:
            with self.subTest(change=change), self.assertRaises(CorruptImageError):
                Superblock.parse(super_with(change), 131072)

    def test_64bit_descriptor_sizes(self):
        """Проверить допустимый размер дескриптора при 64bit."""
        for size in (32, 96, 2048):
            data = super_with((0x60, '<I', 0xC2), (0xFE, '<H', size))
            with self.subTest(size=size), self.assertRaises(CorruptImageError):
                Superblock.parse(data, 131072)
        value = Superblock.parse(super_with((0x60, '<I', 0xC2), (0xFE, '<H', 64)), 131072)
        self.assertEqual(value.descriptor_size, 64)

    def test_explicit_checksum_seed(self):
        """Прочитать явно записанную основу CRC метаданных."""
        value = Superblock.parse(super_with((0x60, '<I', 0x2042), (0x270, '<I', 0x12345678)), 131072)
        self.assertEqual(value.checksum_seed, 0x12345678)

    def test_group_count_partial_last_group(self):
        """Учесть неполную последнюю группу блоков."""
        value = Superblock.parse(super_with((4, '<I', 200), (0x20, '<I', 128), (0, '<I', 32)), 200 * 1024)
        self.assertEqual(value.groups_count, 2)

    def test_inode_count_cannot_leave_last_group_empty(self):
        """Согласовать число inode с количеством групп."""
        data = super_with((4, '<I', 200), (0x20, '<I', 128))
        with self.assertRaisesRegex(CorruptImageError, 'Количество'):
            Superblock.parse(data, 200 * 1024)

    def test_label_stops_at_nul_and_escapes_invalid_utf8(self):
        """Обрезать метку по NUL и экранировать неверную UTF-8."""
        data = bytearray(super_bytes())
        data[0x78:0x88] = b'lab\xff\0ignored'.ljust(16, b'\0')
        self.assertEqual(Superblock.parse(data, 131072).label, 'lab\\xff')

    def test_inode_invalid_extra_size(self):
        """Отклонить невозможный размер расширенных полей inode."""
        for size in (1, 2, 3, 129):
            data = bytearray(256)
            struct.pack_into('<H', data, 0x80, size)
            with self.subTest(size=size), self.assertRaises(CorruptImageError):
                Inode.parse(12, data)

    def test_inode_invalid_nanoseconds(self):
        """Отклонить наносекундную часть вне диапазона."""
        data = bytearray(256)
        struct.pack_into('<H', data, 0x80, 12)
        struct.pack_into('<I', data, 0x88, 1_000_000_000 << 2)
        with self.assertRaisesRegex(CorruptImageError, 'дата'):
            Inode.parse(12, data)

    def test_old_inode_signed_time(self):
        """Прочитать отрицательную знаковую дату прежнего inode."""
        data = bytearray(128)
        struct.pack_into('<i', data, 0x10, -1)
        self.assertEqual(Inode.parse(12, data).mtime_ns, -1_000_000_000)

    def test_unadvertised_extra_fields_ignored(self):
        """Не использовать дополнительные поля без объявления их размера."""
        data = bytearray(256)
        struct.pack_into('<I', data, 0x88, 0xFFFFFFFF)
        self.assertEqual(Inode.parse(12, data).mtime_ns, 0)

    def test_all_inode_kinds(self):
        """Сопоставить все типы UNIX русским обозначениям."""
        expected = {stat.S_IFREG: 'файл', stat.S_IFDIR: 'каталог', stat.S_IFLNK: 'ссылка',
                    stat.S_IFIFO: 'FIFO', stat.S_IFCHR: 'символьное устройство',
                    stat.S_IFBLK: 'блочное устройство', stat.S_IFSOCK: 'сокет', 0: 'неизвестный'}
        for mode, name in expected.items():
            data = bytearray(128)
            struct.pack_into('<H', data, 0, mode)
            with self.subTest(mode=mode):
                self.assertEqual(Inode.parse(12, data).kind, name)

    def test_symlink_does_not_use_regular_file_size_high(self):
        """Не расширять размер ссылки полем обычного файла."""
        data = bytearray(128)
        struct.pack_into('<H', data, 0, stat.S_IFLNK | 0o777)
        struct.pack_into('<I', data, 4, 7)
        struct.pack_into('<I', data, 0x6C, 0xFFFFFFFF)
        self.assertEqual(Inode.parse(12, data).size, 7)
