"""Проверка границ raw-потока, смещения и освобождения входных ресурсов."""

import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from support import super_bytes
from ext4py.errors import CorruptImageError, Ext4Error
from ext4py.image import Image, open_image


class ImageTests(unittest.TestCase):

    """Проверка адресных границ и времени жизни входного потока."""

    def test_negative_partition_offset(self):
        """Отклонить отрицательное начало раздела."""
        with self.assertRaisesRegex(Ext4Error, 'отрицательным'):
            Image(io.BytesIO(bytes(4096)), -1)

    def test_too_short_image(self):
        """Отклонить поток меньше минимального размера метаданных."""
        for size in (0, 1, 1024, 2047):
            with self.subTest(size=size), self.assertRaises(CorruptImageError):
                Image(io.BytesIO(bytes(size)))

    def test_offset_outside_stream(self):
        """Отклонить начало раздела за пределами входного потока."""
        with self.assertRaises(CorruptImageError):
            Image(io.BytesIO(bytes(4096)), 4097)

    def test_exact_minimum_size(self):
        """Принять точный минимальный размер представления Image."""
        image = Image(io.BytesIO(bytes(2048)))
        self.assertEqual(image.size, 2048)
        self.assertEqual(image.read_at(0, 2048), bytes(2048))

    def test_empty_read_at_end(self):
        """Допустить нулевое чтение на конечной границе."""
        image = Image(io.BytesIO(bytes(2048)))
        self.assertEqual(image.read_at(2048, 0), b'')

    def test_invalid_read_ranges(self):
        """Отклонить отрицательные и выходящие за границы диапазоны."""
        image = Image(io.BytesIO(bytes(2048)))
        for offset, length in ((-1, 1), (0, -1), (2048, 1), (2049, 0), (1024, 1025)):
            with self.subTest(offset=offset, length=length), self.assertRaises(CorruptImageError):
                image.read_at(offset, length)

    def test_out_of_order_reads(self):
        """Поддержать чтение в произвольном порядке с префиксом раздела."""
        content = bytes(range(256)) * 16
        image = Image(io.BytesIO(b'prefix' + content), 6)
        for offset in (3000, 1, 2048, 0, 4095):
            with self.subTest(offset=offset):
                self.assertEqual(image.read_at(offset, 1), content[offset:offset + 1])

    def test_stream_truncated_after_opening(self):
        """Обнаружить усечение источника после определения его длины."""
        stream = io.BytesIO(bytes(4096))
        image = Image(stream)
        stream.truncate(2048)
        with self.assertRaisesRegex(CorruptImageError, 'усечён'):
            image.read_at(3000, 1)

    def test_file_closed_after_context(self):
        """Закрыть входной файл по окончании open_image."""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'image.raw'
            path.write_bytes(bytes(1024) + super_bytes())
            with open_image(str(path), io.BytesIO()) as image:
                self.assertFalse(image.stream.closed)
            self.assertTrue(image.stream.closed)

    def test_stdin_temporary_closed_on_exception(self):
        """Удалить временный stdin-файл при исключении внутри with."""
        with self.assertRaisesRegex(RuntimeError, 'failure'):
            with open_image('-', io.BytesIO(bytes(4096))) as image:
                stream = image.stream
                raise RuntimeError('failure')
        self.assertTrue(stream.closed)

    def test_failed_stdin_copy_cleans_temporary(self):
        """Закрыть временный файл при ошибке сохранения stdin."""
        temporary = tempfile.TemporaryFile()
        with patch('ext4py.image.tempfile.TemporaryFile', return_value=temporary), \
                patch('ext4py.image.shutil.copyfileobj', side_effect=OSError('no space')):
            with self.assertRaisesRegex(OSError, 'no space'):
                with open_image('-', io.BytesIO()) as image:
                    self.fail('Не должен возвращать частичный образ')
        self.assertTrue(temporary.closed)

    def test_missing_image_file(self):
        """Передать ошибку открытия отсутствующего образа."""
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(FileNotFoundError):
                with open_image(str(Path(directory) / 'absent'), io.BytesIO()):
                    self.fail('Не должен открывать отсутствующий файл')
