"""Внутрипроцессные проверки CLI с изолированными двоичными потоками."""

import argparse
from contextlib import ExitStack
import io
import os
from pathlib import Path
import runpy
import sys
import tarfile
import tempfile
import unittest
from unittest.mock import patch

from support import path_fixture
from ext4py import cli
from ext4py.errors import Ext4Error


class CliTests(unittest.TestCase):

    """Операции и отказы CLI с независимыми потоками и временным образом."""

    def setUp(self):
        """Создать независимую фикстуру перед каждым сценарием."""
        self.workspace = tempfile.TemporaryDirectory()
        self.addCleanup(self.workspace.cleanup)
        self.directory = Path(self.workspace.name)
        self.image = self.directory / 'disk.raw'
        self.image.write_bytes(path_fixture().data)

    def invoke(self, arguments, data=b'', dispatch_error=None, broken_pipe_success=False):
        """Запустить cli.main с изолированными стандартными потоками.

        Args:
            arguments: Аргументы без имени программы.
            data: Байты stdin для режима образа '-'.
            dispatch_error: Исключение вместо выполнения операции либо None.
            broken_pipe_success: Подменить системное перенаправление stdout,
                не затрагивая настоящий вывод процесса unittest.

        Returns:
            Тройку (код завершения, байты stdout, текст stderr).

        SystemExit преобразуется в код; неожиданные исключения передаются тесту.
        """
        with ExitStack() as stack:
            input_stream = stack.enter_context(io.TextIOWrapper(io.BytesIO(data), encoding='utf-8'))
            output = stack.enter_context(io.TextIOWrapper(io.BytesIO(), encoding='utf-8'))
            error = io.StringIO()
            stack.enter_context(patch('sys.stdin', input_stream))
            stack.enter_context(patch('sys.stdout', output))
            stack.enter_context(patch('sys.stderr', error))
            if dispatch_error is not None:
                stack.enter_context(patch('ext4py.cli.dispatch', side_effect=dispatch_error))
            if broken_pipe_success:
                stack.enter_context(patch.object(output, 'fileno', return_value=99))
                stack.enter_context(patch('ext4py.cli.os.open', return_value=42))
                stack.enter_context(patch('ext4py.cli.os.dup2', return_value=99))
                self.close_mock = stack.enter_context(patch('ext4py.cli.os.close'))
            try:
                status = cli.main([str(arg) for arg in arguments])
            except SystemExit as exit_error:
                status = exit_error.code
            output.flush()
            return status, output.buffer.getvalue(), error.getvalue()

    def test_ls_has_required_metadata_without_flags(self):
        """Вывести обязательные метаданные ls без дополнительных флагов."""
        status, output, error = self.invoke([self.image, 'ls', '/'])
        self.assertEqual(status, 0, error)
        self.assertIn(b'13 1970-01-01 00:00:00 UTC', output)
        self.assertIn('файл: file'.encode(), output)
        self.assertIn('ссылка: rel -> dir/child'.encode(), output)

    def test_ls_single_file(self):
        """Показать метаданные одного файла вместо обхода каталога."""
        status, output, error = self.invoke([self.image, 'ls', '/file'])
        self.assertEqual(status, 0, error)
        self.assertEqual(len(output.splitlines()), 1)
        self.assertIn('файл: /file'.encode(), output)

    def test_ls_all_includes_dot_entries(self):
        """Включить служебные точки при ls -a."""
        status, output, error = self.invoke([self.image, 'ls', '-a'])
        self.assertEqual(status, 0, error)
        self.assertIn('каталог: .\n'.encode(), output)
        self.assertIn('каталог: ..\n'.encode(), output)

    def test_ls_byte_order_is_independent_of_locale(self):
        """Сортировать вывод по исходным байтам имён."""
        status, output, error = self.invoke([self.image, 'ls'])
        self.assertEqual(status, 0, error)
        names = [line.decode().split(': ', 1)[1].split(' -> ')[0] for line in output.splitlines()]
        self.assertEqual(names, sorted(names, key=lambda name: name.encode()))

    def test_cat_preserves_binary_bytes(self):
        """Передать все байтовые значения cat без перекодирования."""
        fixture = path_fixture().put(12, 0o100644, bytes(range(256)) * 3)
        self.image.write_bytes(fixture.data)
        status, output, error = self.invoke([self.image, 'cat', '/file'])
        self.assertEqual(status, 0, error)
        self.assertEqual(output, bytes(range(256)) * 3)
        self.assertEqual(error, '')

    def test_cat_stdin_with_offset(self):
        """Прочитать файл через stdin с начальным смещением."""
        data = bytes(4096) + self.image.read_bytes()
        status, output, error = self.invoke(['--offset', '0x1000', '-', 'cat', '/file'], data)
        self.assertEqual(status, 0, error)
        self.assertEqual(output, b'expected-data')

    def test_cp_and_tar_use_current_directory(self):
        """Сохранить файл и архив с исходными именами в cwd."""
        previous = Path.cwd()
        try:
            os.chdir(self.directory)
            status, output, error = self.invoke([self.image, 'cp', '/file'])
            self.assertEqual(status, 0, error)
            self.assertEqual(output, b'')
            self.assertEqual((self.directory / 'file').read_bytes(), b'expected-data')
            status, output, error = self.invoke([self.image, 'tar', '/dir'])
            self.assertEqual(status, 0, error)
            self.assertEqual(output, b'')
            with tarfile.open(self.directory / 'dir.tar.gz') as archive:
                self.assertEqual(archive.extractfile('dir/child').read(), b'expected-data')
        finally:
            os.chdir(previous)

    def test_help_does_not_require_existing_image(self):
        """Показать общую справку до открытия образа."""
        status, output, error = self.invoke(['--help'])
        self.assertEqual(status, 0, error)
        self.assertIn(b'{ls,cat,cp,tar}', output)
        self.assertNotIn(b'--allow-unclean', output)

    def test_command_help(self):
        """Показать справку каждой из четырёх команд."""
        for command in ('ls', 'cat', 'cp', 'tar'):
            with self.subTest(command=command):
                status, output, error = self.invoke(['absent.raw', command, '--help'])
                self.assertEqual(status, 0, error)
                self.assertIn(command.encode(), output)

    def test_version_without_image(self):
        """Показать версию без обязательного имени образа."""
        status, output, error = self.invoke(['--version'])
        self.assertEqual(status, 0, error)
        self.assertRegex(output.decode(), r'^ext4py \d+\.\d+\.\d+\n$')

    def test_argument_errors(self):
        """Вернуть код 2 и stderr для ошибочного синтаксиса."""
        for arguments in ([], [self.image], [self.image, 'unknown'], [self.image, 'cat'],
                          [self.image, 'cat', '/file', '/dir'], ['--offset', 'x', self.image, 'ls']):
            with self.subTest(arguments=arguments):
                status, output, error = self.invoke(arguments)
                self.assertEqual(status, 2)
                self.assertEqual(output, b'')
                self.assertIn('ошибка аргументов', error)

    def test_missing_image_is_reported_without_traceback(self):
        """Сообщить об отсутствующем образе без traceback."""
        status, output, error = self.invoke([self.directory / 'absent.raw', 'ls'])
        self.assertEqual(status, 1)
        self.assertEqual(output, b'')
        self.assertIn('ext4py: ошибка:', error)
        self.assertNotIn('Traceback', error)

    def test_corrupt_image_is_reported(self):
        """Показать ошибку сигнатуры повреждённого образа."""
        self.image.write_bytes(bytes(4096))
        status, output, error = self.invoke([self.image, 'ls'])
        self.assertEqual(status, 1)
        self.assertIn('Сигнатура', error)

    def test_expected_runtime_errors(self):
        """Преобразовать предусмотренные ошибки операции в код 1."""
        for exception in (Ext4Error('bad data'), OSError('read error'),
                          ValueError('bad value'), tarfile.StreamError('bad archive')):
            with self.subTest(exception=type(exception).__name__):
                status, output, error = self.invoke([self.image, 'ls'], dispatch_error=exception)
                self.assertEqual(status, 1)
                self.assertIn(str(exception), error)
                self.assertNotIn('Traceback', error)

    def test_keyboard_interrupt(self):
        """Преобразовать прерывание пользователем в код 130."""
        status, output, error = self.invoke([self.image, 'ls'], dispatch_error=KeyboardInterrupt())
        self.assertEqual(status, 130)
        self.assertIn('прервана', error)

    def test_broken_pipe_redirects_output(self):
        """Закрыть временный дескриптор после перенаправления вывода."""
        status, output, error = self.invoke([self.image, 'ls'], dispatch_error=BrokenPipeError(), broken_pipe_success=True)
        self.assertEqual(status, 1)
        self.assertEqual(error, '')
        self.close_mock.assert_called_once_with(42)

    def test_broken_pipe_without_available_fileno(self):
        """Обработать закрытый pipe без доступного fileno."""
        status, output, error = self.invoke([self.image, 'ls'], dispatch_error=BrokenPipeError())
        self.assertEqual(status, 1)
        self.assertEqual(error, '')

    def test_broken_pipe_failure_closes_descriptor(self):
        """Закрыть открытый дескриптор и при ошибке перенаправления."""
        with patch('ext4py.cli.os.open', return_value=42), patch('ext4py.cli.os.close') as close:
            status, output, error = self.invoke([self.image, 'ls'], dispatch_error=BrokenPipeError())
        self.assertEqual(status, 1)
        close.assert_called_once_with(42)

    def test_unexpected_programming_errors_are_not_suppressed(self):
        """Передать вызывающему коду неожиданную программную ошибку."""
        with self.assertRaisesRegex(RuntimeError, 'bug'):
            self.invoke([self.image, 'ls'], dispatch_error=RuntimeError('bug'))

    def test_module_entrypoint(self):
        """Запустить справку через модульную точку входа."""
        output = io.StringIO()
        with patch.object(sys, 'argv', ['ext4py', '--help']), patch('sys.stdout', output):
            with self.assertRaises(SystemExit) as result:
                runpy.run_module('ext4py', run_name='__main__')
        self.assertEqual(result.exception.code, 0)
        self.assertIn('{ls,cat,cp,tar}', output.getvalue())


class CliFormattingTests(unittest.TestCase):

    """Чистые преобразования аргументов, имён и временных меток."""

    def test_offset_decimal_and_hex(self):
        """Принять десятичные и шестнадцатеричные смещения."""
        for value, expected in [('0', 0), ('001024', 1024), ('0x1000', 4096), ('0XFF', 255)]:
            with self.subTest(value=value):
                self.assertEqual(cli.parse_offset(value), expected)

    def test_invalid_offsets(self):
        """Отклонить пустые, отрицательные и нецелые смещения."""
        for value in ('', '-1', '1.2', '0x', 'junk'):
            with self.subTest(value=value), self.assertRaises(argparse.ArgumentTypeError):
                cli.parse_offset(value)

    def test_safe_text_leaves_unicode_readable(self):
        """Сохранить печатные русские имена без лишнего экранирования."""
        self.assertEqual(cli.safe_text('отчёт с пробелом.txt'), 'отчёт с пробелом.txt')

    def test_safe_text_escapes_control_characters(self):
        """Экранировать управляющие символы и недекодируемые байты."""
        for text in ('a\nb', '\x1b[31m', 'a\tb', 'name\udcff'):
            with self.subTest(text=text):
                result = cli.safe_text(text)
                self.assertTrue(result.isprintable())
                self.assertNotIn('\x1b', result)

    def test_timestamp_epoch_and_nanoseconds(self):
        """Точно показать эпоху, наносекунды и отрицательное время."""
        self.assertEqual(cli.format_time(0), '1970-01-01 00:00:00 UTC')
        self.assertEqual(cli.format_time(1), '1970-01-01 00:00:00.000000001 UTC')
        self.assertEqual(cli.format_time(-1), '1969-12-31 23:59:59.999999999 UTC')

    def test_timestamp_outside_datetime_range(self):
        """Вернуть числовое время вне диапазона datetime."""
        value = 10**30
        self.assertEqual(cli.format_time(value), f'{value // 10**9} с от эпохи Unix')
