"""Проверка записей каталогов, имён, длин и контрольных сумм."""

from dataclasses import replace
import struct
import unittest
from unittest.mock import patch

from support import RawFixture, directory_block, inode
from ext4py.errors import CorruptImageError
from ext4py.structures import crc32c


class DirectoryTests(unittest.TestCase):

    """Корректные и повреждённые записи каталогов разных форматов."""

    def parse(self, data, filetype=True):
        """Разобрать переданный блок каталога через реальный файловый слой.

        Args:
            data: Содержимое каталога, включая намеренные повреждения.
            filetype: Выбор новой или прежней длины имени.

        Returns:
            Список DirectoryEntry после полного обхода.
        """
        fixture = RawFixture().put(2, 0o40755, data)
        fs = fixture.filesystem()
        if not filetype:
            fs.sb = replace(fs.sb, incompat=fs.sb.incompat & ~2)
        return list(fs.directory(fs.inode(2)))

    def test_directory_name_bytes_round_trip(self):
        """Сохранить исходные недекодируемые байты имени."""
        name = b'file-\xff'
        result = self.parse(directory_block([(12, name)]))
        self.assertEqual(result[0].name.encode('utf-8', 'surrogateescape'), name)

    def test_unicode_name(self):
        """Декодировать русское имя каталожной записи."""
        name = 'отчёт.txt'
        self.assertEqual(self.parse(directory_block([(12, name.encode())]))[0].name, name)

    def test_maximum_name_length(self):
        """Принять имя предельной длины 255 байт."""
        self.assertEqual(len(self.parse(directory_block([(12, b'a' * 255)]))[0].name), 255)

    def test_old_directory_format(self):
        """Разобрать прежнее 16-битное поле длины имени."""
        self.assertEqual(self.parse(directory_block([(12, b'old')], filetype=False), False)[0].name, 'old')

    def test_unused_record_skipped(self):
        """Пропустить незанятую запись, сохранив соседнюю живую."""
        entries = self.parse(directory_block([(0, b'unused'), (12, b'live')]))
        self.assertEqual([(e.inode, e.name) for e in entries], [(12, 'live')])

    def test_invalid_record_lengths(self):
        """Отклонить нулевые, короткие и невыровненные длины записей."""
        for size in (0, 4, 10, 1028):
            data = bytearray(directory_block([(12, b'x')]))
            struct.pack_into('<H', data, 4, size)
            with self.subTest(size=size), self.assertRaisesRegex(CorruptImageError, 'длина записи'):
                self.parse(data)

    def test_truncated_record_header(self):
        """Обнаружить остаток блока короче заголовка записи."""
        data = bytearray(directory_block([(12, b'x')]))
        struct.pack_into('<H', data, 4, 1020)
        with self.assertRaisesRegex(CorruptImageError, 'Усечён заголовок'):
            self.parse(data)

    def test_live_record_requires_name(self):
        """Не допустить занятую запись с пустым именем."""
        data = bytearray(directory_block([(12, b'x')]))
        data[6] = 0
        with self.assertRaisesRegex(CorruptImageError, 'длина имени'):
            self.parse(data)

    def test_name_must_fit_record(self):
        """Проверить вместимость имени в пределах записи."""
        data = bytearray(directory_block([(12, b'x')]))
        struct.pack_into('<H', data, 4, 8)
        with self.assertRaisesRegex(CorruptImageError, 'длина имени'):
            self.parse(data)

    def test_legacy_name_length_limit(self):
        """Отклонить имя длиннее лимита в старом формате."""
        data = bytearray(directory_block([(12, b'x')], filetype=False))
        struct.pack_into('<H', data, 6, 256)
        with self.assertRaisesRegex(CorruptImageError, 'длина имени'):
            self.parse(data, False)

    def test_name_components_cannot_escape_directory(self):
        """Отклонить NUL и разделитель пути внутри имени."""
        for name in (b'a/b', b'a\0b'):
            with self.subTest(name=name), self.assertRaisesRegex(CorruptImageError, 'Некорректное имя'):
                self.parse(directory_block([(12, name)]))

    def test_inode_number_limit(self):
        """Не допустить ссылку каталога на inode за границей таблиц."""
        with self.assertRaisesRegex(CorruptImageError, 'номер inode'):
            self.parse(directory_block([(33, b'x')]))

    def test_64k_directory_record_encoding(self):
        """Разобрать специальные длины записи полного блока 64 КиБ."""
        for encoded in (0, 65535):
            fixture = RawFixture()
            fs = fixture.filesystem()
            fs.sb = replace(fs.sb, block_size=65536)
            data = bytearray(directory_block([(12, b'large')], block_size=65536))
            struct.pack_into('<H', data, 4, encoded)
            value = replace(inode(bytes(60)), mode=0o40755, size=65536)
            with patch.object(fs, 'iter_data', return_value=iter([bytes(data)])):
                self.assertEqual(list(fs.directory(value))[0].name, 'large')

    def test_leaf_checksum_valid_and_damaged(self):
        """Проверить CRC листа каталога и обнаружить его повреждение."""
        fixture = RawFixture()
        data = bytearray(directory_block([(12, b'checked')]))
        struct.pack_into('<H', data, 4, 1012)
        data[-12:] = struct.pack('<IHBBI', 0, 12, 0, 0xDE, 0)
        fs = fixture.filesystem()
        seed = crc32c(struct.pack('<II', 2, 0), fs.sb.checksum_seed)
        struct.pack_into('<I', data, 1020, crc32c(data[:-12], seed))
        fixture.put(2, 0o40755, data)
        fs = fixture.filesystem()
        node = fs.inode(2)
        fs.sb = replace(fs.sb, ro_compat=0x400)
        self.assertEqual(list(fs.directory(node))[0].name, 'checked')
        fixture.data[20 * 1024 + 8] ^= 1
        broken = fixture.filesystem()
        node = broken.inode(2)
        broken.sb = replace(broken.sb, ro_compat=0x400)
        with self.assertRaisesRegex(CorruptImageError, 'контрольная сумма'):
            list(broken.directory(node))
