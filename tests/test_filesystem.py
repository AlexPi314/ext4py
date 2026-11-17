"""Проверка inode, путей, ссылок и адаптера потокового чтения."""

from dataclasses import replace
import io
import stat
import struct
import unittest
from unittest.mock import patch

from support import directory_block, path_fixture
from ext4py.errors import CorruptImageError, Ext4Error, UnsupportedFeatureError
from ext4py.filesystem import InodeReader


class FileSystemTests(unittest.TestCase):

    """Доступ к inode, проверка данных и разрешение путей в raw-фикстуре."""

    def setUp(self):
        """Создать независимую фикстуру перед каждым сценарием."""
        self.fixture = path_fixture()
        self.fs = self.fixture.filesystem()

    def test_read_block_bounds(self):
        """Отклонить недопустимые номера физических блоков."""
        for number in (-1, 0, 128):
            with self.subTest(number=number), self.assertRaises(CorruptImageError):
                self.fs.read_block(number)

    def test_inode_number_bounds(self):
        """Проверить нижнюю и верхнюю границы номера inode."""
        for number in (-1, 0, 33):
            with self.subTest(number=number), self.assertRaises(CorruptImageError):
                self.fs.inode(number)

    def test_group_number_bounds(self):
        """Проверить диапазон номеров групп."""
        for number in (-1, 1):
            with self.subTest(number=number), self.assertRaises(CorruptImageError):
                self.fs._inode_table(number)

    def test_inode_table_bounds(self):
        """Отклонить таблицу inode за объявленными границами EXT4."""
        for block in (0, 127):
            struct.pack_into('<I', self.fixture.data, 2048 + 8, block)
            with self.subTest(block=block), self.assertRaisesRegex(CorruptImageError, 'Таблица'):
                self.fixture.filesystem().inode(2)

    def test_inode_without_links_refused(self):
        """Отклонить переход к inode без жёстких ссылок."""
        self.fixture.put(12, 0o100644, b'x', links=0)
        with self.assertRaisesRegex(CorruptImageError, 'неиспользуемый'):
            self.fixture.filesystem().inode(12)

    def test_inode_without_type_refused(self):
        """Отклонить узел без указанного типа объекта."""
        self.fixture.put(12, 0, b'x')
        with self.assertRaises(CorruptImageError):
            self.fixture.filesystem().inode(12)

    def test_group_descriptor_cached(self):
        """Загрузить дескриптор одной группы только при первом обращении."""
        with patch.object(self.fs.image, 'read_at', wraps=self.fs.image.read_at) as read:
            self.fs.inode(2)
            self.fs.inode(3)
        self.assertEqual(sum(call.args[0] == 2048 for call in read.call_args_list), 1)

    def test_chunk_size_is_positive(self):
        """Отклонить неположительный размер порции чтения."""
        for size in (0, -1):
            with self.subTest(size=size), self.assertRaises(ValueError):
                list(self.fs.iter_data(self.fs.inode(12), size))

    def test_data_flags_refused(self):
        """Отклонить неподдерживаемые флаги хранения данных inode."""
        for flag in (4, 0x100, 0x200, 0x800, 0x10000000, 0x40000000):
            value = replace(self.fs.inode(12), flags=flag)
            with self.subTest(flag=flag), self.assertRaises(UnsupportedFeatureError):
                list(self.fs.iter_data(value))

    def test_special_file_data_refused(self):
        """Не читать специальный объект как последовательность обычных данных."""
        for mode in (stat.S_IFIFO, stat.S_IFCHR, stat.S_IFBLK, stat.S_IFSOCK):
            value = replace(self.fs.inode(12), mode=mode)
            with self.subTest(mode=mode), self.assertRaises(Ext4Error):
                list(self.fs.iter_data(value))

    def test_last_block_is_trimmed(self):
        """Обрезать последний блок и соблюдать максимальный размер порции."""
        value = self.fs.inode(12)
        chunks = list(self.fs.iter_data(value, 4))
        self.assertEqual(b''.join(chunks), b'expected-data')
        self.assertTrue(all(len(chunk) <= 4 for chunk in chunks))

    def test_empty_file_has_no_chunks(self):
        """Не выдавать порции содержимого пустого файла."""
        self.fixture.put(17, 0o100644)
        fs = self.fixture.filesystem()
        self.assertEqual(list(fs.iter_data(fs.inode(17))), [])

    def test_reading_directory_requires_directory_type(self):
        """Отклонить обычный файл в операции чтения каталога."""
        with self.assertRaisesRegex(Ext4Error, 'не каталог'):
            list(self.fs.directory(self.fs.inode(12)))

    def test_directory_size_alignment(self):
        """Проверить кратность размера каталога блоку."""
        value = replace(self.fs.inode(2), size=1023)
        with self.assertRaisesRegex(CorruptImageError, 'не кратен'):
            list(self.fs.directory(value))

    def test_root_and_repeated_separators(self):
        """Разрешить корень, повторные разделители и точку."""
        self.assertEqual(self.fs.resolve('////').number, 2)
        self.assertEqual(self.fs.resolve('/dir//./child').number, 12)

    def test_relative_path_refused(self):
        """Отклонить относительный путь и нулевой байт."""
        for path in ('', 'dir/child', './file', '../file', '/file\0extra'):
            with self.subTest(path=path), self.assertRaises(Ext4Error):
                self.fs.resolve(path)

    def test_absolute_and_relative_symlinks(self):
        """Разрешить оба способа задания цели ссылки."""
        self.assertEqual(self.fs.resolve('/rel').number, 12)
        self.assertEqual(self.fs.resolve('/abs').number, 12)

    def test_do_not_follow_final_link(self):
        """Вернуть последний inode ссылки при отключённом разыменовании."""
        self.assertEqual(self.fs.resolve('/rel', False).number, 13)

    def test_root_symlink(self):
        """Разрешить ссылку на корень и путь через неё."""
        self.assertEqual(self.fs.resolve('/rootlink').number, 2)
        self.assertEqual(self.fs.resolve('/rootlink/file').number, 12)

    def test_symlink_target_trailing_slash(self):
        """Потребовать каталог для цели ссылки с завершающим слешем."""
        self.assertEqual(self.fs.resolve('/dirlink').number, 3)
        self.assertEqual(self.fs.resolve('/dirlink/child').number, 12)

    def test_parent_after_symlink(self):
        """Обработать '..' после ссылки и ограничить переход выше корня."""
        self.assertEqual(self.fs.resolve('/dirlink/../file').number, 12)
        self.assertEqual(self.fs.resolve('/../../file').number, 12)

    def test_missing_component(self):
        """Сообщить об отсутствующем компоненте пути."""
        with self.assertRaisesRegex(Ext4Error, 'не найден'):
            self.fs.resolve('/dir/absent')

    def test_file_in_intermediate_component(self):
        """Не использовать обычный файл как промежуточный каталог."""
        with self.assertRaisesRegex(Ext4Error, 'не является каталогом'):
            self.fs.resolve('/file/child')

    def test_trailing_slash_requires_directory(self):
        """Отклонить завершающий слеш после обычного файла."""
        with self.assertRaisesRegex(Ext4Error, 'каталог'):
            self.fs.resolve('/file/')

    def test_duplicate_directory_names(self):
        """Обнаружить неоднозначное повторение имени при поиске пути."""
        self.fixture.put(2, 0o40755, directory_block([(12, b'file'), (13, b'file')]))
        with self.assertRaisesRegex(CorruptImageError, 'Повторяющееся'):
            self.fixture.filesystem().resolve('/file')

    def test_exact_symlink_transition_limit(self):
        """Принять ровно 40 переходов и отклонить превышение."""
        with patch.object(self.fs, 'readlink', side_effect=['rel'] * 39 + ['file']):
            self.assertEqual(self.fs.resolve('/rel').number, 12)
        with patch.object(self.fs, 'readlink', return_value='rel'):
            with self.assertRaisesRegex(Ext4Error, '40'):
                self.fs.resolve('/rel')

    def test_readlink_non_link_refused(self):
        """Отклонить чтение цели у обычного файла."""
        with self.assertRaises(Ext4Error):
            self.fs.readlink(self.fs.inode(12))

    def test_readlink_invalid_size(self):
        """Проверить нулевую и чрезмерную длину цели ссылки."""
        value = self.fs.inode(13)
        for size in (0, 4097):
            with self.subTest(size=size), self.assertRaises(CorruptImageError):
                self.fs.readlink(replace(value, size=size))

    def test_short_link_cannot_exceed_inline_capacity(self):
        """Не читать короткую ссылку за пределами 60 байт i_block."""
        with self.assertRaisesRegex(CorruptImageError, 'не помещается'):
            self.fs.readlink(replace(self.fs.inode(13), size=61))

    def test_link_nul_refused(self):
        """Отклонить нулевой байт в цели ссылки."""
        self.fixture.put(17, 0o120777, b'a\0b', fast=True)
        fs = self.fixture.filesystem()
        with self.assertRaisesRegex(CorruptImageError, 'Нулевой'):
            fs.readlink(fs.inode(17))

    def test_short_link_with_external_attributes(self):
        """Учесть блок атрибутов при распознавании короткой ссылки."""
        value = replace(self.fs.inode(13), file_acl=9, blocks=2)
        self.assertEqual(self.fs.readlink(value), 'dir/child')

    def test_block_link(self):
        """Прочитать длинную блочную ссылку и пройти по её цели."""
        target = b'dir/' + b'./' * 40 + b'child'
        self.fixture.put(17, 0o120777, target)
        self.fixture.put(2, 0o40755, directory_block([(3, b'dir'), (17, b'long')]))
        fs = self.fixture.filesystem()
        self.assertEqual(fs.readlink(fs.inode(17)), target.decode())
        self.assertEqual(fs.resolve('/long').number, 12)

    def test_open_file_rejects_other_types(self):
        """Ограничить open_file обычными файлами."""
        for number in (2, 13):
            with self.subTest(number=number), self.assertRaises(Ext4Error):
                self.fs.open_file(self.fs.inode(number))

    def test_open_file_context_closes_stream(self):
        """Закрыть поток файла при выходе из контекста."""
        with self.fs.open_file(self.fs.inode(12)) as stream:
            self.assertEqual(stream.read(), b'expected-data')
        self.assertTrue(stream.closed)


class ReaderTests(unittest.TestCase):

    """Контракт RawIOBase при границах порций, закрытии и отказах источника."""

    def test_readinto_across_chunk_boundaries(self):
        """Заполнить буфер несколькими порциями с сохранением остатка."""
        reader = InodeReader(iter([b'ab', b'cde', b'f']))
        buffer = bytearray(4)
        self.assertEqual(reader.readinto(buffer), 4)
        self.assertEqual(buffer, b'abcd')
        self.assertEqual(reader.readinto(buffer), 2)
        self.assertEqual(buffer[:2], b'ef')
        self.assertEqual(reader.readinto(buffer), 0)

    def test_empty_chunks_do_not_end_stream(self):
        """Пропустить пустые порции до настоящего конца итератора."""
        with io.BufferedReader(InodeReader(iter([b'', b'ab', b'', b'cd', b'']))) as stream:
            self.assertEqual(stream.read(), b'abcd')

    def test_empty_destination_does_not_consume_input(self):
        """Не потреблять источник при буфере нулевой длины."""
        reader = InodeReader(iter([b'abc']))
        self.assertEqual(reader.readinto(bytearray()), 0)
        self.assertEqual(reader.read(), b'abc')

    def test_read_after_close_refused(self):
        """Отклонить чтение закрытого адаптера."""
        reader = InodeReader(iter([b'abc']))
        reader.close()
        with self.assertRaises(ValueError):
            reader.readinto(bytearray(1))

    def test_buffered_partial_reads(self):
        """Согласовать частичные чтения с BufferedReader."""
        with io.BufferedReader(InodeReader(iter([b'abc', b'def'])), buffer_size=2) as stream:
            self.assertEqual(stream.read(1), b'a')
            self.assertEqual(stream.read(3), b'bcd')
            self.assertEqual(stream.read(), b'ef')

    def test_iterator_failure_propagates(self):
        """Передать ошибку генератора вызывающему читателю."""
        def chunks():
            """Выдать один байт, затем передать читателю CorruptImageError источника."""
            yield b'a'
            raise CorruptImageError('damaged')
        reader = InodeReader(chunks())
        with self.assertRaisesRegex(CorruptImageError, 'damaged'):
            reader.readinto(bytearray(2))
