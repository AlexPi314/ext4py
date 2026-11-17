"""Эталонные CRC32C, геометрия и расширенные метаданные inode."""

import io
import struct
import unittest

from support import super_bytes
from ext4py.errors import CorruptImageError, UnsupportedFeatureError
from ext4py.image import Image, open_image
from ext4py.structures import Inode, Superblock, crc32c


class StructureTests(unittest.TestCase):

    """Контрольные векторы и исходные сценарии разбора структур."""

    def test_crc32c_known_vector(self):
        """Сопоставить CRC32C с независимым известным вектором."""
        self.assertEqual(crc32c(b'123456789') ^ 0xFFFFFFFF, 0xE3069283)

    def test_superblock_geometry_and_bounds(self):
        """Принять базовую геометрию и отклонить её повреждения и усечение."""
        self.assertEqual(Superblock.parse(super_bytes(), 131072).block_size, 1024)
        for offset in (0, 4, 0x20, 0x28):
            data = bytearray(super_bytes())
            struct.pack_into('<I', data, offset, 0)
            with self.subTest(offset=offset), self.assertRaises(CorruptImageError):
                Superblock.parse(data, 131072)
        with self.assertRaises(CorruptImageError):
            Superblock.parse(super_bytes(), 4096)

    def test_unsupported_and_unclean(self):
        """Не принимать неподдерживаемые или несогласованные структуры."""
        data = bytearray(super_bytes())
        struct.pack_into('<I', data, 0x60, 0x8000)
        with self.assertRaises(UnsupportedFeatureError):
            Superblock.parse(data, 131072)
        data = bytearray(super_bytes())
        struct.pack_into('<I', data, 0x60, 0x46)
        with self.assertRaises(CorruptImageError):
            Superblock.parse(data, 131072)

    def test_superblock_checksum(self):
        """Проверить корректную CRC суперблока и обнаружить изменение."""
        data = bytearray(super_bytes())
        struct.pack_into('<I', data, 0x64, 0x400)
        data[0x175] = 1
        struct.pack_into('<I', data, 0x3FC, crc32c(data[:0x3FC]))
        Superblock.parse(data, 131072)
        data[0x78] ^= 1
        with self.assertRaisesRegex(CorruptImageError, 'контрольная сумма'):
            Superblock.parse(data, 131072)

    def test_inode_extended_timestamp_and_ids(self):
        """Объединить старшие UID/GID, размер и точное время после 2038 года."""
        data = bytearray(256)
        struct.pack_into('<H', data, 0, 0o100644)
        struct.pack_into('<I', data, 4, 42)
        struct.pack_into('<I', data, 0x6C, 1)
        struct.pack_into('<i', data, 0x10, -0x80000000)
        struct.pack_into('<H', data, 0x80, 32)
        struct.pack_into('<I', data, 0x88, (123456789 << 2) | 1)
        struct.pack_into('<H', data, 0x78, 2)
        struct.pack_into('<H', data, 0x7A, 3)
        value = Inode.parse(12, data)
        self.assertEqual(value.mtime_ns, 2147483648 * 10**9 + 123456789)
        self.assertEqual((value.size, value.uid, value.gid), ((1 << 32) + 42, 2 << 16, 3 << 16))

    def test_image_offset_and_stdin(self):
        """Одинаково читать смещённый файл и сохранённый stdin с проверкой границ."""
        image = Image(io.BytesIO(b'prefix' + b'x' * 2048), 6)
        self.assertEqual(image.read_at(0, 3), b'xxx')
        with self.assertRaises(CorruptImageError):
            image.read_at(2047, 2)
        with open_image('-', io.BytesIO(b'x' * 2048)) as piped:
            self.assertEqual(piped.read_at(2047, 1), b'x')


if __name__ == '__main__':
    unittest.main()
