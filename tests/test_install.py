"""Проверка установки и очистки собственных файлов при сбоях.

Все пути расположены во временном prefix; окружение пользователя
и существующие установки не изменяются.
"""

import importlib.util
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

SCRIPT = Path(__file__).resolve().parents[1] / 'scripts/install.py'
SPEC = importlib.util.spec_from_file_location('ext4py_install', SCRIPT)
installer = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(installer)


class InstallTests(unittest.TestCase):
    """Успешная установка и очистка только файлов текущей попытки."""

    def setUp(self):
        """Создать независимую фикстуру перед каждым сценарием."""
        self.workspace = tempfile.TemporaryDirectory()
        self.addCleanup(self.workspace.cleanup)
        self.prefix = Path(self.workspace.name) / 'prefix'
        self.application = self.prefix / 'lib/ext4py'
        self.launcher = self.prefix / 'bin/ext4py'

    def test_installed_launcher_runs(self):
        """Запустить установленную точку входа и проверить состав пакета."""
        self.assertEqual(installer.install(self.prefix), self.launcher)
        result = subprocess.run([sys.executable, str(self.launcher), '--version'], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertRegex(result.stdout, r'^ext4py \d+\.\d+\.\d+\n$')
        self.assertTrue((self.application / 'ext4py/cli.py').is_file())
        self.assertTrue(self.launcher.stat().st_mode & 0o111)

    def test_existing_installation_is_unchanged(self):
        """Отказаться от повторной установки без изменения прежнего launcher."""
        installer.install(self.prefix)
        original = self.launcher.read_bytes()
        with self.assertRaises(FileExistsError):
            installer.install(self.prefix)
        self.assertEqual(self.launcher.read_bytes(), original)

    def test_existing_broken_launcher_symlink_is_preserved(self):
        """Сохранить существующую оборванную ссылку launcher."""
        self.launcher.parent.mkdir(parents=True)
        self.launcher.symlink_to('missing')
        with self.assertRaises(FileExistsError):
            installer.install(self.prefix)
        self.assertTrue(self.launcher.is_symlink())
        self.assertFalse(self.application.exists())

    def test_partial_copy_is_removed(self):
        """Удалить частично скопированное приложение при ошибке copytree."""
        def fail_copy(source, destination, **kwargs):
            """Имитировать copytree: создать partial.py в destination и вызвать OSError.

            source и kwargs сохраняют сигнатуру копировщика; результат не возвращается.
            """
            destination = Path(destination)
            destination.mkdir(parents=True)
            (destination / 'partial.py').write_bytes(b'partial')
            raise OSError('copy failure')
        with patch.object(installer.shutil, 'copytree', side_effect=fail_copy):
            with self.assertRaisesRegex(OSError, 'copy failure'):
                installer.install(self.prefix)
        self.assertFalse(self.application.exists())
        self.assertFalse(self.launcher.exists())

    def test_launcher_write_failure_removes_application(self):
        """Очистить приложение после отказа создания launcher."""
        original_open = Path.open
        def fail_launcher(path, *args, **kwargs):
            """Передать path, args и kwargs исходному Path.open, кроме пути launcher.

            Для launcher вызвать OSError до создания файла; иначе вернуть поток.
            """
            if path == self.launcher:
                raise OSError('write failure')
            return original_open(path, *args, **kwargs)
        with patch.object(Path, 'open', fail_launcher):
            with self.assertRaisesRegex(OSError, 'write failure'):
                installer.install(self.prefix)
        self.assertFalse(self.application.exists())
        self.assertFalse(self.launcher.exists())

    def test_chmod_failure_removes_both_outputs(self):
        """Удалить созданные файлы при невозможности назначить права."""
        with patch.object(Path, 'chmod', side_effect=OSError('chmod failure')):
            with self.assertRaisesRegex(OSError, 'chmod failure'):
                installer.install(self.prefix)
        self.assertFalse(self.application.exists())
        self.assertFalse(self.launcher.exists())

    def test_concurrent_launcher_is_not_deleted(self):
        """Сохранить чужой launcher, появившийся во время установки."""
        original_open = Path.open
        def create_competitor(path, *args, **kwargs):
            """Для path launcher создать чужой файл и вызвать FileExistsError.

            Для остальных путей вернуть исходный Path.open с args и kwargs.
            """
            if path == self.launcher:
                with original_open(path, 'wb') as output:
                    output.write(b'other installation')
                raise FileExistsError('concurrent launcher')
            return original_open(path, *args, **kwargs)
        with patch.object(Path, 'open', create_competitor):
            with self.assertRaises(FileExistsError):
                installer.install(self.prefix)
        self.assertFalse(self.application.exists())
        self.assertEqual(self.launcher.read_bytes(), b'other installation')

    def test_copy_interruption_is_cleaned(self):
        """Очистить прерванную копию после KeyboardInterrupt."""
        def interrupt(source, destination, **kwargs):
            """Создать destination, затем имитировать прерывание copytree.

            source и kwargs совместимы с копировщиком; функция вызывает KeyboardInterrupt.
            """
            Path(destination).mkdir(parents=True)
            raise KeyboardInterrupt
        with patch.object(installer.shutil, 'copytree', side_effect=interrupt):
            with self.assertRaises(KeyboardInterrupt):
                installer.install(self.prefix)
        self.assertFalse(self.application.exists())


if __name__ == '__main__':
    unittest.main()
