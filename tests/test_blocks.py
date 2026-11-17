"""Базовые сценарии адресации, разреженности и повреждённых карт блоков."""

import io
import struct
import unittest

from support import extent_node, inode, superblock
from ext4py.blocks import BlockMap
from ext4py.errors import CorruptImageError
from ext4py.filesystem import FileSystem
from ext4py.image import Image


class BlockTests(unittest.TestCase):

    """Базовые сценарии extent, косвенной адресации и разреженных данных."""

    def test_extent_length_32768_is_initialized(self):
        """Считать длину 32768 инициализированной, а 32769 — unwritten длины 1."""
        from dataclasses import replace
        sb = replace(superblock(), blocks_count=100000)
        value = inode(extent_node([(0, 32768, 10), (32768, 32769, 40000)]), 32769 * 1024)
        self.assertEqual(list(BlockMap(sb, lambda n: b'', value).ranges()),
                         [(0, 32768, 10), (32768, 1, None)])

    def test_multilevel_extent_tree(self):
        """Пройти два уровня индексов до листового extent."""
        blocks = {10: extent_node([(0, 11, 0)], depth=1, size=1024),
                  11: extent_node([(0, 2, 20)], size=1024)}
        value = inode(extent_node([(0, 10, 0)], depth=2), size=2048)
        self.assertEqual(list(BlockMap(superblock(), blocks.__getitem__, value).ranges()), [(0, 2, 20)])

    def test_bad_extent_trees(self):
        """Отклонить перекрытия, циклы и неправильную глубину дерева."""
        cases = [
            (extent_node([(0, 4, 20), (2, 1, 30)]), {}),
            (extent_node([(0, 10, 0)], depth=2), {10: extent_node([(0, 10, 0)], depth=1, size=1024)}),
            (extent_node([(0, 10, 0)], depth=2), {10: extent_node([(0, 1, 20)], size=1024)}),
        ]
        for root, blocks in cases:
            with self.subTest(root=root), self.assertRaises(CorruptImageError):
                list(BlockMap(superblock(), blocks.__getitem__, inode(root)).ranges())

    def test_sparse_and_unwritten_data(self):
        """Заполнить нулями разреженные и unwritten-области."""
        from support import super_bytes
        raw = bytearray(128 * 1024)
        raw[1024:2048] = super_bytes()
        raw[20 * 1024:21 * 1024] = b'A' * 1024
        raw[21 * 1024:22 * 1024] = b'S' * 1024
        fs = FileSystem(Image(io.BytesIO(raw)))
        value = inode(extent_node([(1, 1, 20), (3, 32769, 21)]), size=4 * 1024 + 7)
        actual = b''.join(fs.iter_data(value, chunk_size=257))
        self.assertEqual(actual, bytes(1024) + b'A' * 1024 + bytes(2 * 1024 + 7))

    def test_all_legacy_indirection_levels(self):
        """Дойти до тройного косвенного указателя без разворачивания пустых ветвей."""
        pointers = [0] * 15
        pointers[14] = 10
        blocks = {}
        for current, child in ((10, 11), (11, 12), (12, 20)):
            data = bytearray(1024)
            struct.pack_into('<I', data, 0, child)
            blocks[current] = bytes(data)
        logical = 12 + 256 + 256**2
        value = inode(struct.pack('<15I', *pointers), size=(logical + 1) * 1024, flags=0)
        ranges = list(BlockMap(superblock(), blocks.__getitem__, value).ranges())
        self.assertEqual(ranges[-1], (logical, 1, 20))
        self.assertLess(len(ranges), 20)

    def test_directory_bad_record_length(self):
        """Завершить повреждённый каталог ошибкой без бесконечного обхода."""
        from dataclasses import replace
        from support import super_bytes
        raw = bytearray(128 * 1024)
        raw[1024:2048] = super_bytes()
        fs = FileSystem(Image(io.BytesIO(raw)))
        value = replace(inode(extent_node([(0, 1, 20)])), mode=0o40755)
        with self.assertRaises(CorruptImageError):
            list(fs.directory(value))


if __name__ == '__main__':
    unittest.main()
