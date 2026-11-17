"""Проверка выгрузки, атомарной публикации, отказов и заголовков tar."""

from dataclasses import replace
import io
from pathlib import Path
import stat
import struct
import tarfile
import tempfile
import unittest
from unittest.mock import patch

from support import inode
from support import path_fixture
from ext4py.errors import CorruptImageError, Ext4Error
from ext4py.export import atomic_output, copy_file, export_name, export_tar, write_tar, _tar_info
from ext4py.filesystem import DirectoryEntry


class ExportTests(unittest.TestCase):

    """Сохранность файлов назначения и точность архивных записей."""

    def setUp(self):
        """Создать независимую фикстуру перед каждым сценарием."""
        self.fs = path_fixture().filesystem()
        self.workspace = tempfile.TemporaryDirectory()
        self.addCleanup(self.workspace.cleanup)
        self.directory = Path(self.workspace.name)

    def test_export_basename_rules(self):
        """Выбрать безопасное имя результата для корня, точек и обычного пути."""
        for path, expected in [('/', 'root'), ('////', 'root'), ('/a/b/', 'b'),
                               ('/a/.', 'root'), ('/a/..', 'root'), ('/a/отчёт.txt', 'отчёт.txt')]:
            with self.subTest(path=path):
                self.assertEqual(export_name(path), expected)

    def test_atomic_output_content(self):
        """Публиковать имя только после полной записи содержимого."""
        target = self.directory / 'result'
        with atomic_output(target) as output:
            output.write(b'complete')
            self.assertFalse(target.exists())
        self.assertEqual(target.read_bytes(), b'complete')
        self.assertEqual([p.name for p in self.directory.iterdir()], ['result'])

    def test_atomic_output_removes_partial_file(self):
        """Удалить временную запись после исключения в with."""
        target = self.directory / 'result'
        with self.assertRaisesRegex(RuntimeError, 'failure'):
            with atomic_output(target) as output:
                output.write(b'partial')
                raise RuntimeError('failure')
        self.assertEqual(list(self.directory.iterdir()), [])

    def test_existing_file_not_modified(self):
        """Сохранить прежние байты существующего назначения."""
        target = self.directory / 'result'
        target.write_bytes(b'original')
        with self.assertRaises(Ext4Error):
            with atomic_output(target):
                self.fail('Не должен перезаписывать файл')
        self.assertEqual(target.read_bytes(), b'original')

    def test_broken_destination_symlink_not_followed(self):
        """Не следовать оборванной ссылке назначения."""
        target = self.directory / 'result'
        target.symlink_to(self.directory / 'absent')
        with self.assertRaises(Ext4Error):
            with atomic_output(target):
                self.fail('Не должен следовать ссылке')
        self.assertFalse((self.directory / 'absent').exists())

    def test_destination_created_during_export(self):
        """Сохранить файл, появившийся во время выгрузки."""
        target = self.directory / 'result'
        with self.assertRaises(FileExistsError):
            with atomic_output(target) as output:
                output.write(b'export')
                target.write_bytes(b'concurrent')
        self.assertEqual(target.read_bytes(), b'concurrent')
        self.assertEqual(len(list(self.directory.iterdir())), 1)

    def test_link_failure_cleans_temporary(self):
        """Очистить временный файл после отказа os.link."""
        with patch('ext4py.export.os.link', side_effect=OSError('link unavailable')):
            with self.assertRaisesRegex(OSError, 'unavailable'):
                with atomic_output(self.directory / 'result') as output:
                    output.write(b'data')
        self.assertEqual(list(self.directory.iterdir()), [])

    def test_flush_failure_cleans_temporary(self):
        """Очистить временный файл после отказа fsync."""
        with patch('ext4py.export.os.fsync', side_effect=OSError('disk error')):
            with self.assertRaisesRegex(OSError, 'disk error'):
                with atomic_output(self.directory / 'result') as output:
                    output.write(b'data')
        self.assertEqual(list(self.directory.iterdir()), [])

    def test_missing_destination_parent(self):
        """Не создавать отсутствующую иерархию назначения неявно."""
        with self.assertRaises(FileNotFoundError):
            with atomic_output(self.directory / 'missing/result'):
                self.fail('Не должен создавать отсутствующий каталог')

    def test_copy_uses_source_basename_in_destination_directory(self):
        """Сохранить имя исходного пути при заданном каталоге назначения."""
        target = copy_file(self.fs, '/rel', str(self.directory))
        self.assertEqual(target.name, 'rel')
        self.assertEqual(target.read_bytes(), b'expected-data')

    def test_copy_read_failure_cleans_partial_output(self):
        """Удалить незавершённый результат при сбое чтения файла."""
        def chunks():
            """Выдать первую порцию файла и затем имитировать повреждение блока.

            Yields:
                Байты частичного чтения перед CorruptImageError.
            """
            yield b'partial'
            raise CorruptImageError('bad block')
        with patch.object(self.fs, 'resolve', return_value=self.fs.inode(12)), \
                patch.object(self.fs, 'iter_data', return_value=chunks()):
            with self.assertRaisesRegex(CorruptImageError, 'bad block'):
                copy_file(self.fs, '/file', str(self.directory / 'result'))
        self.assertEqual(list(self.directory.iterdir()), [])

    def test_copy_directory_refused(self):
        """Отклонить попытку копирования каталога как обычного файла."""
        with self.assertRaises(Ext4Error):
            copy_file(self.fs, '/dir', str(self.directory / 'result'))
        self.assertEqual(list(self.directory.iterdir()), [])

    def test_archive_source_must_be_directory(self):
        """Отклонить файловый источник для экспорта каталога."""
        with self.assertRaisesRegex(Ext4Error, 'каталогу'):
            export_tar(self.fs, '/file', str(self.directory / 'result.tar.gz'))
        self.assertEqual(list(self.directory.iterdir()), [])

    def test_archive_directory_cycle(self):
        """Обнаружить ссылку на каталог-предок при архивировании."""
        with patch.object(self.fs, 'directory', return_value=iter([DirectoryEntry('cycle', 2)])):
            with self.assertRaisesRegex(CorruptImageError, 'Цикл'):
                export_tar(self.fs, '/', str(self.directory / 'result.tar.gz'))
        self.assertEqual(list(self.directory.iterdir()), [])

    def test_archive_duplicate_names(self):
        """Отклонить повторяющиеся имена внутри архивируемого каталога."""
        with patch.object(self.fs, 'directory', return_value=iter([DirectoryEntry('x', 12), DirectoryEntry('x', 12)])):
            with self.assertRaisesRegex(CorruptImageError, 'Повторяющееся'):
                export_tar(self.fs, '/', str(self.directory / 'result.tar.gz'))
        self.assertEqual(list(self.directory.iterdir()), [])

    def test_archive_retains_empty_directories(self):
        """Сохранить пустой каталог отдельным заголовком tar."""
        from support import RawFixture, directory_block
        fs = RawFixture().put(2, 0o40755, directory_block([(2, b'.'), (2, b'..')])).filesystem()
        output = io.BytesIO()
        write_tar(fs, '/', output)
        with tarfile.open(fileobj=io.BytesIO(output.getvalue()), mode='r:gz') as archive:
            self.assertEqual(archive.getnames(), ['root'])
            self.assertTrue(archive.getmember('root').isdir())

    def test_tar_header_preserves_metadata(self):
        """Перенести UID, GID, права и размер в заголовок tar."""
        node = replace(self.fs.inode(12), uid=123, gid=456, mode=stat.S_IFREG | 0o6754)
        info = _tar_info(self.fs, node, 'name')
        self.assertEqual((info.uid, info.gid, info.mode, info.size), (123, 456, 0o6754, 13))

    def test_tar_nanoseconds_are_exact_decimal(self):
        """Записать знаковое наносекундное время точной строкой PAX."""
        for instant, expected in [(1, '0.000000001'), (-1, '-0.000000001'),
                                  (1_123_456_789, '1.123456789')]:
            node = replace(self.fs.inode(12), mtime_ns=instant)
            with self.subTest(instant=instant):
                self.assertEqual(_tar_info(self.fs, node, 'name').pax_headers['mtime'], expected)

    def test_tar_special_types(self):
        """Представить каталог и FIFO соответствующими типами tar."""
        for mode, kind in [(stat.S_IFDIR, tarfile.DIRTYPE), (stat.S_IFIFO, tarfile.FIFOTYPE)]:
            with self.subTest(mode=mode):
                self.assertEqual(_tar_info(self.fs, replace(self.fs.inode(12), mode=mode), 'x').type, kind)

    def test_tar_old_device_number(self):
        """Декодировать прежнее представление номеров устройства."""
        node = replace(inode(struct.pack('<I', 0x1234)), mode=stat.S_IFCHR)
        info = _tar_info(self.fs, node, 'device')
        self.assertEqual((info.type, info.devmajor, info.devminor), (tarfile.CHRTYPE, 0x12, 0x34))

    def test_tar_new_device_number(self):
        """Декодировать новое представление major и minor."""
        major, minor = 0xABC, 0x54321
        encoded = (minor & 255) | (major << 8) | ((minor & ~255) << 12)
        node = replace(inode(struct.pack('<II', 0, encoded)), mode=stat.S_IFBLK)
        info = _tar_info(self.fs, node, 'device')
        self.assertEqual((info.type, info.devmajor, info.devminor), (tarfile.BLKTYPE, major, minor))

    def test_tar_socket_refused(self):
        """Завершить экспорт сокета явной ошибкой."""
        with self.assertRaisesRegex(Ext4Error, 'сокет'):
            _tar_info(self.fs, replace(self.fs.inode(12), mode=stat.S_IFSOCK), 'socket')
