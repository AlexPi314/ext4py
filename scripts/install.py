#!/usr/bin/env python3
"""Необязательная установка исходников ext4py в пользовательский prefix без pip."""

import argparse
from contextlib import suppress
from pathlib import Path
import shutil
import sys


def install(prefix: Path) -> Path:
    """Установить пакет и launcher в новый пользовательский prefix.

    Args:
        prefix: Корень установки с подкаталогами lib и bin.

    Returns:
        Путь созданной исполняемой точки входа.

    Raises:
        FileExistsError: Установка или имя launcher уже существуют.
        OSError: Копирование, запись или назначение прав завершилось ошибкой.

    При отказе удаляются только файлы текущей попытки. Прерывание также
    инициирует очистку; ошибки удаления OSError не подменяют исходную причину.
    """
    source = Path(__file__).resolve().parents[1] / 'src' / 'ext4py'
    application = prefix / 'lib' / 'ext4py'
    launcher = prefix / 'bin' / 'ext4py'
    if application.exists() or application.is_symlink() or launcher.exists() or launcher.is_symlink():
        raise FileExistsError('Установка уже существует; выберите другой --prefix')
    application.parent.mkdir(parents=True, exist_ok=True)
    launcher.parent.mkdir(parents=True, exist_ok=True)
    application.mkdir()
    # При сбое можно удалять только launcher, созданный текущей попыткой.
    launcher_created = False
    try:
        shutil.copytree(source, application / 'ext4py', ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
        with launcher.open('x', encoding='utf-8') as output:
            launcher_created = True
            output.write('#!/usr/bin/env python3\n'
                         '"""Запуск установленного пакета ext4py."""\n'
                         'import sys\n'
                         f'sys.path.insert(0, {str(application)!r})\n'
                         'from ext4py.cli import main\n'
                         'raise SystemExit(main())\n')
        launcher.chmod(0o755)
    except BaseException:
        shutil.rmtree(application, ignore_errors=True)
        if launcher_created:
            with suppress(OSError):
                launcher.unlink(missing_ok=True)
        raise
    return launcher


def main() -> int:
    """Разобрать --prefix, выполнить установку и сообщить её путь.

    Returns:
        0 при успехе или 1 при ошибке хоста.

    Argparse завершает справку или ошибку аргументов через SystemExit.
    """
    parser = argparse.ArgumentParser(description="Установка ext4py без pip и сборочных зависимостей")
    parser.add_argument('--prefix', type=Path, default=Path.home() / '.local',
                        help='каталог установки (по умолчанию ~/.local)')
    args = parser.parse_args()
    if sys.version_info < (3, 10):
        parser.error('требуется Python 3.10 или новее')
    try:
        launcher = install(args.prefix.expanduser().resolve())
    except OSError as error:
        print(f'Ошибка установки: {error}', file=sys.stderr)
        return 1
    print(f'Установлено: {launcher}\nДобавьте {launcher.parent} в PATH, если требуется.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
