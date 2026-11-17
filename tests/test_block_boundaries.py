"""Границы extent-дерева, адресов и косвенных указателей."""

from dataclasses import replace
import struct
import unittest

from support import extent_node, inode, superblock
from ext4py.blocks import BlockMap
from ext4py.errors import CorruptImageError
from ext4py.structures import crc32c


class BlockBoundaryTests(unittest.TestCase):

    """Граничные карты блоков с проверкой адресов, глубины и CRC."""

    def map(self, root, blocks=None, size=1024, sb=None, flags=0x80000):
        """Получить список диапазонов из заданной синтетической карты.

        Args:
            root: Байты поля i_block.
            blocks: Соответствие адресов внешним блокам; None означает пустой словарь.
            size: Логический размер файла в байтах.
            sb: Геометрия либо None для базовой фикстуры.
            flags: Флаги выбора extent или косвенных указателей.

        Returns:
            Список BlockRange; ошибки разбора не перехватываются.
        """
        values = {} if blocks is None else blocks
        return list(BlockMap(sb or superblock(), values.__getitem__, inode(root, size, flags)).ranges())

    def test_empty_extent_file(self):
        """Проверить отсутствие диапазонов у пустого extent-файла."""
        self.assertEqual(self.map(extent_node([]), size=0), [])

    def test_bad_extent_magic(self):
        """Отклонить узел с неверной сигнатурой extent."""
        root = bytearray(extent_node([]))
        root[0] = 0
        with self.assertRaisesRegex(CorruptImageError, 'заголовок'):
            self.map(root)

    def test_invalid_extent_header_counts(self):
        """Отклонить невозможные вместимость, число записей и глубину."""
        for offset, value in ((2, 5), (4, 0), (4, 5), (6, 6)):
            root = bytearray(extent_node([]))
            struct.pack_into('<H', root, offset, value)
            with self.subTest(offset=offset, value=value), self.assertRaises(CorruptImageError):
                self.map(root)

    def test_empty_index_node_refused(self):
        """Отклонить индексный узел без дочерних записей."""
        with self.assertRaises(CorruptImageError):
            self.map(extent_node([], depth=1))

    def test_duplicate_extent_keys(self):
        """Отклонить повторяющиеся логические ключи extent."""
        with self.assertRaisesRegex(CorruptImageError, 'порядок'):
            self.map(extent_node([(0, 1, 20), (0, 1, 30)]))

    def test_decreasing_extent_keys(self):
        """Отклонить убывающие ключи дерева."""
        with self.assertRaisesRegex(CorruptImageError, 'порядок'):
            self.map(extent_node([(2, 1, 20), (1, 1, 30)]))

    def test_zero_extent_length(self):
        """Отклонить extent нулевой длины."""
        with self.assertRaisesRegex(CorruptImageError, 'длина'):
            self.map(extent_node([(0, 0, 20)]))

    def test_zero_physical_extent(self):
        """Отклонить нулевой адрес выделенного extent."""
        with self.assertRaisesRegex(CorruptImageError, 'адрес блока'):
            self.map(extent_node([(0, 1, 0)]))

    def test_physical_extent_past_image(self):
        """Отклонить физический диапазон за границей образа."""
        with self.assertRaises(CorruptImageError):
            self.map(extent_node([(0, 2, 127)]))

    def test_last_valid_physical_block(self):
        """Прочитать диапазон на последнем допустимом физическом блоке."""
        self.assertEqual(self.map(extent_node([(0, 1, 127)])), [(0, 1, 127)])

    def test_48bit_extent_address(self):
        """Объединить старшую и младшую части 48-битного адреса."""
        sb = replace(superblock(), blocks_count=(1 << 33) + 100)
        self.assertEqual(self.map(extent_node([(0, 1, (1 << 32) + 7)]), sb=sb),
                         [(0, 1, (1 << 32) + 7)])

    def test_child_extent_outside_parent_interval(self):
        """Отклонить листовой диапазон за интервалом родителя."""
        blocks = {10: extent_node([(0, 3, 20)], size=1024),
                  11: extent_node([(2, 1, 30)], size=1024)}
        with self.assertRaisesRegex(CorruptImageError, 'длина'):
            self.map(extent_node([(0, 10, 0), (2, 11, 0)], depth=1), blocks)

    def test_child_key_before_parent_lower_bound(self):
        """Отклонить ключ до нижней границы поддерева."""
        blocks = {10: extent_node([(1, 1, 20)], size=1024)}
        with self.assertRaisesRegex(CorruptImageError, 'Перекрытие'):
            self.map(extent_node([(2, 10, 0)], depth=1), blocks)

    def test_multiple_leaf_nodes_in_logical_order(self):
        """Выдать диапазоны нескольких листьев в логическом порядке."""
        blocks = {10: extent_node([(0, 1, 20)], size=1024),
                  11: extent_node([(4, 2, 30)], size=1024)}
        self.assertEqual(self.map(extent_node([(0, 10, 0), (4, 11, 0)], depth=1), blocks),
                         [(0, 1, 20), (4, 2, 30)])

    def test_extent_node_checksum(self):
        """Принять корректную CRC узла и обнаружить изменение его байтов."""
        sb = replace(superblock(), ro_compat=0x400)
        node = bytearray(extent_node([(0, 1, 20)], size=1024))
        seed = crc32c(struct.pack('<II', 12, 0), sb.checksum_seed)
        struct.pack_into('<I', node, 1020, crc32c(node[:1020], seed))
        root = extent_node([(0, 10, 0)], depth=1)
        self.assertEqual(self.map(root, {10: bytes(node)}, sb=sb), [(0, 1, 20)])
        node[30] ^= 1
        with self.assertRaisesRegex(CorruptImageError, 'контрольная сумма'):
            self.map(root, {10: bytes(node)}, sb=sb)

    def test_direct_blocks_stop_at_eof(self):
        """Ограничить прямую адресацию логическим концом файла."""
        pointers = [20, 21, 22] + [0] * 12
        self.assertEqual(self.map(struct.pack('<15I', *pointers), size=1025, flags=0),
                         [(0, 1, 20), (1, 1, 21)])

    def test_legacy_empty_file(self):
        """Не выдавать прямые или косвенные диапазоны пустого файла."""
        self.assertEqual(self.map(bytes(60), size=0, flags=0), [])

    def test_invalid_direct_block(self):
        """Отклонить прямой указатель за пределами EXT4."""
        pointers = [128] + [0] * 14
        with self.assertRaises(CorruptImageError):
            self.map(struct.pack('<15I', *pointers), flags=0)

    def test_indirect_cycle(self):
        """Обнаружить цикл в косвенных блоках."""
        pointers = [0] * 15
        pointers[13] = 10
        child = struct.pack('<I', 10) + bytes(1020)
        with self.assertRaisesRegex(CorruptImageError, 'Цикл'):
            self.map(struct.pack('<15I', *pointers), {10: child}, size=(12 + 256 + 1) * 1024, flags=0)

    def test_legacy_size_beyond_address_capacity(self):
        """Отклонить размер больше ёмкости старой карты указателей."""
        capacity = 12 + 256 + 256**2 + 256**3
        with self.assertRaisesRegex(CorruptImageError, 'ёмкость'):
            self.map(bytes(60), size=(capacity + 1) * 1024, flags=0)

    def test_single_indirect_data_and_hole(self):
        """Различать данные и нулевую область в косвенном блоке."""
        pointers = [0] * 15
        pointers[12] = 10
        child = struct.pack('<II', 20, 0) + bytes(1016)
        result = self.map(struct.pack('<15I', *pointers), {10: child}, size=14 * 1024, flags=0)
        self.assertEqual(result[-2:], [(12, 1, 20), (13, 1, None)])
