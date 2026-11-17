"""Интеграционные проверки на восьми конфигурациях реальных EXT4-образов.

Нужен mke2fs; при наличии e2fsck проверяется HTree.
Права администратора и монтирование не используются.
"""

from contextlib import contextmanager
import hashlib
import io
import os
from pathlib import Path
import shutil
import stat
import struct
import subprocess
import sys
import tarfile
import tempfile
import unittest

import support
from ext4py.errors import CorruptImageError, Ext4Error
from ext4py.export import copy_file, export_tar
from ext4py.filesystem import FileSystem
from ext4py.image import Image, open_image
from ext4py.structures import crc32c

ROOT = Path(__file__).resolve().parents[1]


@unittest.skipUnless(shutil.which('mke2fs'), 'Для интеграционных тестов требуется mke2fs из e2fsprogs')
class IntegrationTests(unittest.TestCase):

    """Чтение реальных образов и запуск CLI без монтирования или root."""

    @classmethod
    def setUpClass(cls):
        """Создать общий стенд восьми EXT4-образов через mke2fs и при наличии e2fsck.

        Файлы известного дерева формируются до разбора парсером; root не требуется.
        """
        cls.workspace = tempfile.TemporaryDirectory(prefix='ext4py-test-')
        cls.base = Path(cls.workspace.name)
        cls.tree = cls.base / 'tree'
        (cls.tree / 'docs' / 'empty-dir').mkdir(parents=True)
        (cls.tree / 'docs' / 'пример.txt').write_text('Тестирование EXT4\n', encoding='utf-8')
        (cls.tree / 'binary').write_bytes(bytes(range(256)) * 4097)
        (cls.tree / 'empty').touch()
        (cls.tree / '.hidden').write_bytes(b'hidden')
        (cls.tree / 'docs' / 'line\nbreak').write_bytes(b'control-name')
        (cls.tree / 'docs' / 'relative').symlink_to('../binary')
        (cls.tree / 'absolute').symlink_to('/docs/пример.txt')
        (cls.tree / 'dirlink').symlink_to('docs')
        (cls.tree / 'loop').symlink_to('loop')
        (cls.tree / 'dangling').symlink_to('/absent')
        (cls.tree / 'longlink').symlink_to('docs/' + './' * 40 + 'пример.txt')
        os.link(cls.tree / 'binary', cls.tree / 'hard')
        os.mkfifo(cls.tree / 'fifo')
        with (cls.tree / 'sparse').open('wb') as stream:
            stream.write(b'BEGIN')
            stream.seek(3 * 1024 * 1024 + 3)
            stream.write(b'END')
        with (cls.tree / 'fragmented').open('wb') as stream:
            for index in range(100):
                stream.seek(index * 8192)
                stream.write(bytes([index + 1]) * 4096)
        (cls.tree / 'many').mkdir()
        for index in range(400):
            (cls.tree / 'many' / f'file-{index:04d}').write_bytes(str(index).encode())
        cls.images = []
        for name, block_size, features in [('ext4-1k', '1024', []), ('ext4-2k', '2048', []),
                                           ('ext4-4k', '4096', []), ('ext4-64k', '65536', []),
                                           ('ext4-32bit', '1024', ['-O', '^64bit']),
                                           ('legacy', '1024', ['-O', '^extent,^64bit,^metadata_csum']),
                                           ('small-inode', '1024', ['-I', '128', '-O', '^64bit']),
                                           ('checksum-seed', '4096', ['-O', 'metadata_csum_seed'])]:
            image = cls.base / (name + '.raw')
            with image.open('wb') as stream:
                stream.truncate(64 * 1024 * 1024)
            result = subprocess.run(['mke2fs', '-q', '-F', '-t', 'ext4', '-b', block_size,
                                     *features, '-d', str(cls.tree), str(image)], capture_output=True)
            if result.returncode:
                raise RuntimeError(result.stderr.decode(errors='replace'))
            if shutil.which('e2fsck'):
                result = subprocess.run(['e2fsck', '-fyD', str(image)], capture_output=True)
                if result.returncode not in (0, 1):
                    raise RuntimeError(result.stderr.decode(errors='replace'))
            cls.images.append(image)

    @classmethod
    def tearDownClass(cls):
        """Удалить все временные образы и результаты общего интеграционного стенда."""
        cls.workspace.cleanup()

    @contextmanager
    def filesystem(self, image=None):
        """Открыть файловую систему одного стендового образа в контексте.

        Args:
            image: Путь выбранного образа или None для первой конфигурации.

        Yields:
            FileSystem; заимствованный вход закрывается при выходе из with.
        """
        with open_image(str(image or self.images[0]), io.BytesIO()) as image_file:
            yield FileSystem(image_file)

    def cli(self, *args, input=None, cwd=None):
        """Запустить настоящий CLI отдельным процессом с тайм-аутом.

        Args:
            args: Аргументы после имени launcher.
            input: Байты stdin или None.
            cwd: Рабочая директория процесса или None для текущей.

        Returns:
            CompletedProcess с кодом и байтовыми потоками, без автоматической
            ошибки для ненулевого кода, поскольку он может проверяться тестом.
        """
        return subprocess.run([sys.executable, str(ROOT / 'ext4py'), *map(str, args)],
                              input=input, cwd=cwd, capture_output=True, timeout=60)

    def test_file_contents_all_geometries(self):
        """Сопоставить SHA-256 файлов во всех конфигурациях реальных образов."""
        for image in self.images:
            with self.subTest(image=image.name), self.filesystem(image) as fs:
                for name in ('empty', 'binary', 'sparse', 'fragmented', 'docs/пример.txt'):
                    actual = b''.join(fs.iter_data(fs.resolve('/' + name)))
                    self.assertEqual(hashlib.sha256(actual).digest(),
                                     hashlib.sha256((self.tree / name).read_bytes()).digest())

    def test_directory_metadata_and_htree(self):
        """Проверить типы, размеры и полный обход HTree-каталога."""
        for image in self.images:
            with self.subTest(image=image.name), self.filesystem(image) as fs:
                entries = {entry.name: fs.inode(entry.inode) for entry in fs.directory(fs.resolve('/'))}
                self.assertTrue(stat.S_ISFIFO(entries['fifo'].mode))
                self.assertEqual(entries['binary'].size, (self.tree / 'binary').stat().st_size)
                many = fs.resolve('/many')
                if shutil.which('e2fsck') and many.size > fs.sb.block_size:
                    self.assertTrue(many.flags & 0x1000)
                names = {entry.name for entry in fs.directory(many)} - {'.', '..'}
                self.assertEqual(names, {f'file-{index:04d}' for index in range(400)})

    def test_links_and_path_resolution(self):
        """Проверить поиск путей и разных ссылок на реальных образах."""
        for image in self.images:
            with self.subTest(image=image.name), self.filesystem(image) as fs:
                original = fs.resolve('/docs/пример.txt').number
                self.assertEqual(fs.resolve('/absolute').number, original)
                self.assertEqual(fs.resolve('/longlink').number, original)
                self.assertEqual(fs.resolve('/dirlink/../binary').number, fs.resolve('/binary').number)
                self.assertEqual(fs.resolve('/docs/relative').number, fs.resolve('/hard').number)
                self.assertEqual(fs.resolve('/../../').number, 2)
                self.assertEqual(fs.readlink(fs.resolve('/dangling', False)), '/absent')
                for path in ('/loop', '/dangling', '/binary/..', 'relative', '/binary/'):
                    with self.subTest(path=path), self.assertRaises(Ext4Error):
                        fs.resolve(path)

    def test_file_copy_and_no_overwrite(self):
        """Выгрузить байты с исходным именем и запретить перезапись."""
        with tempfile.TemporaryDirectory() as directory, self.filesystem() as fs:
            target = copy_file(fs, '/binary', directory)
            self.assertEqual(target.name, 'binary')
            self.assertEqual(target.read_bytes(), (self.tree / 'binary').read_bytes())
            with self.assertRaises(Ext4Error):
                copy_file(fs, '/binary', directory)
            link = Path(directory) / 'link'
            link.symlink_to(Path(directory) / 'absent')
            with self.assertRaises(Ext4Error):
                copy_file(fs, '/binary', str(link))
            self.assertFalse((Path(directory) / 'absent').exists())

    def test_tar_structure_contents_links_and_fifo(self):
        """Сопоставить дерево, файлы, ссылки и FIFO созданного архива."""
        with tempfile.TemporaryDirectory() as directory, self.filesystem() as fs:
            output = export_tar(fs, '/', str(Path(directory) / 'root.tar.gz'))
            with tarfile.open(output, 'r:gz') as archive:
                self.assertTrue(archive.getmember('root/docs/empty-dir').isdir())
                self.assertEqual(archive.extractfile('root/binary').read(), (self.tree / 'binary').read_bytes())
                self.assertEqual(archive.extractfile('root/hard').read(), (self.tree / 'binary').read_bytes())
                self.assertTrue(archive.getmember('root/hard').islnk())
                self.assertEqual(archive.getmember('root/absolute').linkname, '/docs/пример.txt')
                self.assertTrue(archive.getmember('root/fifo').isfifo())
                self.assertEqual(archive.extractfile('root/docs/line\nbreak').read(), b'control-name')
                self.assertEqual(archive.extractfile('root/sparse').read(), (self.tree / 'sparse').read_bytes())

    def test_cli_listing_cat_and_errors(self):
        """Проверить внешний процесс CLI, метаданные и сообщения об ошибках."""
        result = self.cli(self.images[0], 'ls', '/docs')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('файл: пример.txt'.encode(), result.stdout)
        self.assertIn(b'UTC', result.stdout)
        self.assertIn(b"'line\\nbreak'", result.stdout)
        result = self.cli(self.images[0], 'cat', '/absolute')
        self.assertEqual(result.stdout, (self.tree / 'docs/пример.txt').read_bytes())
        self.assertEqual(result.stderr, b'')
        result = self.cli(self.images[0], 'cat', '/absent')
        self.assertEqual(result.returncode, 1)
        self.assertIn('не найден'.encode(), result.stderr)
        self.assertNotIn(b'Traceback', result.stderr)
        self.assertEqual(self.cli(self.images[0], 'unknown').returncode, 2)
        self.assertEqual(self.cli('--offset', '-1', self.images[0], 'ls', '/').returncode, 2)

    def test_cli_stdin_and_offset(self):
        """Передать реальный stdin, в том числе с префиксом перед EXT4."""
        raw = self.images[0].read_bytes()
        expected = (self.tree / 'docs/пример.txt').read_bytes()
        result = self.cli('-', 'cat', '/absolute', input=raw)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout, expected)
        result = self.cli('--offset', '0x1000', '-', 'cat', '/absolute', input=bytes(4096) + raw)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout, expected)

    def test_cli_default_copy_and_tar(self):
        """Сохранить исходные имена файла и архива в cwd процесса."""
        with tempfile.TemporaryDirectory() as directory:
            result = self.cli(self.images[0], 'cp', '/docs/пример.txt', cwd=directory)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual((Path(directory) / 'пример.txt').read_bytes(),
                             (self.tree / 'docs/пример.txt').read_bytes())
            result = self.cli(self.images[0], 'tar', '/docs', cwd=directory)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertTrue((Path(directory) / 'docs.tar.gz').exists())
            with tarfile.open(Path(directory) / 'docs.tar.gz', mode='r:gz') as archive:
                self.assertEqual(archive.extractfile('docs/пример.txt').read(),
                                 (self.tree / 'docs/пример.txt').read_bytes())

    def test_corrupt_inode_and_directory_checksums(self):
        """Обнаружить повреждение CRC inode и листового блока каталога."""
        raw = self.images[0].read_bytes()
        fs = FileSystem(Image(io.BytesIO(raw)))
        node = fs.resolve('/docs')
        group, index = divmod(node.number - 1, fs.sb.inodes_per_group)
        location = fs._inode_table(group) * fs.sb.block_size + index * fs.sb.inode_size
        damaged = bytearray(raw)
        damaged[location + 4] ^= 1
        with self.assertRaisesRegex(CorruptImageError, 'контрольная сумма inode'):
            FileSystem(Image(io.BytesIO(damaged))).resolve('/docs')
        from ext4py.blocks import BlockMap
        physical = next(BlockMap(fs.sb, fs.read_block, node).ranges())[2]
        damaged = bytearray(raw)
        damaged[physical * fs.sb.block_size + 8] ^= 1
        broken = FileSystem(Image(io.BytesIO(damaged)))
        with self.assertRaisesRegex(CorruptImageError, 'контрольная сумма блока каталога'):
            list(broken.directory(broken.resolve('/docs')))

    def test_truncated_and_unclean_images(self):
        """Отклонить усечённый образ и незавершённое состояние журнала."""
        raw = bytearray(self.images[0].read_bytes())
        with self.assertRaises(CorruptImageError):
            FileSystem(Image(io.BytesIO(raw[:4096])))
        struct.pack_into('<I', raw, 1024 + 0x60, struct.unpack_from('<I', raw, 1024 + 0x60)[0] | 4)
        struct.pack_into('<I', raw, 1024 + 0x3FC, crc32c(raw[1024:1024 + 0x3FC]))
        result = self.cli('-', 'ls', '/', input=raw)
        self.assertEqual(result.returncode, 1)
        self.assertIn('нужен согласованный образ'.encode(), result.stderr)

    def test_minimal_cli_contract(self):
        """Сохранить четыре команды и отвергать удалённые режимы."""
        result = self.cli(self.images[0], 'ls')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('файл: binary'.encode(), result.stdout)
        self.assertIn(b'UTC', result.stdout)
        self.assertNotIn(b'.hidden', result.stdout)
        self.assertIn(b'.hidden', self.cli(self.images[0], 'ls', '-a').stdout)
        for arguments in [('info',), ('stat', '/'), ('readlink', '/absolute'),
                          ('ls', '-l', '/'), ('ls', '-d', '/'),
                          ('ls', '/', '/docs'), ('cat', '/binary', '/empty'),
                          ('cp', '/binary', 'output'), ('tar', '-czf', '-', '/docs')]:
            with self.subTest(arguments=arguments):
                self.assertEqual(self.cli(self.images[0], *arguments).returncode, 2)
        self.assertEqual(self.cli('--allow-unclean', self.images[0], 'ls').returncode, 2)

    def test_group_descriptor_corruption(self):
        """Обнаружить изменение дескриптора группы по CRC."""
        raw = bytearray(self.images[0].read_bytes())
        fs = FileSystem(Image(io.BytesIO(raw)))
        offset = (fs.sb.first_data_block + 1) * fs.sb.block_size
        raw[offset + 12] ^= 1
        with self.assertRaisesRegex(CorruptImageError, 'дескриптора группы'):
            FileSystem(Image(io.BytesIO(raw))).resolve('/')

    def test_external_extent_node_corruption(self):
        """Обнаружить повреждение внешнего узла extent-дерева."""
        from ext4py.structures import u16, u32
        raw = bytearray(self.images[0].read_bytes())
        fs = FileSystem(Image(io.BytesIO(raw)))
        node = fs.resolve('/fragmented')
        self.assertGreater(u16(node.block_data, 6), 0)
        physical = u32(node.block_data, 16) | (u16(node.block_data, 20) << 32)
        raw[physical * fs.sb.block_size + 100] ^= 1
        broken = FileSystem(Image(io.BytesIO(raw)))
        with self.assertRaisesRegex(CorruptImageError, 'узла extent'):
            b''.join(broken.iter_data(broken.resolve('/fragmented')))

    def test_128_byte_inode_checksum_corruption(self):
        """Проверить укороченную CRC записи inode размером 128 байт."""
        image = next(image for image in self.images if image.stem == 'small-inode')
        raw = bytearray(image.read_bytes())
        fs = FileSystem(Image(io.BytesIO(raw)))
        self.assertEqual(fs.sb.inode_size, 128)
        node = fs.resolve('/binary')
        group, index = divmod(node.number - 1, fs.sb.inodes_per_group)
        offset = fs._inode_table(group) * fs.sb.block_size + index * fs.sb.inode_size
        raw[offset + 4] ^= 1
        with self.assertRaisesRegex(CorruptImageError, 'контрольная сумма inode'):
            FileSystem(Image(io.BytesIO(raw))).resolve('/binary')

    def test_failed_directory_archive_leaves_no_output(self):
        """Не оставлять результат после отказа архивирования каталога."""
        with tempfile.TemporaryDirectory() as directory:
            result = self.cli(self.images[0], 'tar', '/absent', cwd=directory)
            self.assertEqual(result.returncode, 1)
            self.assertEqual(list(Path(directory).iterdir()), [])
            self.assertNotIn(b'Traceback', result.stderr)

    def test_broken_output_pipe(self):
        """Завершить внешний cat без traceback после закрытия потребителя."""
        process = subprocess.Popen([sys.executable, str(ROOT / 'ext4py'), str(self.images[0]),
                                    'cat', '/binary'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        process.stdout.close()
        stderr = process.stderr.read()
        self.assertEqual(process.wait(timeout=10), 1)
        process.stderr.close()
        self.assertNotIn(b'Traceback', stderr)


if __name__ == '__main__':
    unittest.main()
