"""Русскоязычный интерфейс четырёх операций чтения EXT4.

Текстовые результаты и байты cat идут в stdout; ошибки и сообщения
о сохранении — в stderr. Общие аргументы предшествуют имени образа.
"""

import argparse
from datetime import datetime, timedelta, timezone
import os
import shutil
import stat
import sys
import tarfile
from typing import Sequence

from . import __version__
from .constants import IO_CHUNK_SIZE
from .errors import Ext4Error
from .export import copy_file, export_tar
from .filesystem import FileSystem
from .image import open_image
from .structures import Inode


class RussianParser(argparse.ArgumentParser):
    """ArgumentParser с русскими заголовками групп и префиксом ошибок."""

    def __init__(self, *args, **kwargs):
        """Настроить группы, передав args и kwargs стандартному ArgumentParser."""
        super().__init__(*args, **kwargs)
        self._positionals.title = "позиционные аргументы"
        self._optionals.title = "параметры"

    def error(self, message: str) -> None:
        """Завершить разбор аргументов с диагностикой и кодом 2.

        Args:
            message: Описание ошибки синтаксиса, полученное от argparse.

        Raises:
            SystemExit: Всегда, после печати синтаксиса и сообщения в stderr.
        """
        self.print_usage(sys.stderr)
        self.exit(2, f"{self.prog}: ошибка аргументов: {message}\n")


def parse_offset(value: str) -> int:
    """Преобразовать десятичное или 0x-смещение в число байт.

    Args:
        value: Текст неотрицательного целого аргумента CLI.

    Returns:
        Начало EXT4 относительно входного файла, в байтах.

    Raises:
        argparse.ArgumentTypeError: Строка не задаёт допустимое смещение.
    """
    try:
        number = int(value, 16 if value.lower().startswith("0x") else 10)
        if number < 0:
            raise ValueError
        return number
    except ValueError as error:
        raise argparse.ArgumentTypeError("ожидается неотрицательное число байт") from error


def make_parser() -> argparse.ArgumentParser:
    """Создать синтаксис ext4py ОБРАЗ КОМАНДА с четырьмя операциями.

    Returns:
        ArgumentParser; аргументы здесь не разбираются и образ не открывается.
    """
    parser = RussianParser(prog="ext4py", description="Чтение raw-образов EXT4 без монтирования.",
                           epilog="Пример: ext4py disk.raw ls /; cat disk.raw | ext4py - cat /etc/hostname")
    parser.add_argument("--version", action="version", version=f"ext4py {__version__}")
    parser.add_argument("--offset", type=parse_offset, default=0, metavar="БАЙТЫ",
                        help="смещение начала раздела EXT4 в raw-образе (по умолчанию 0)")
    parser.add_argument("image", metavar="ОБРАЗ", help="raw-образ либо '-' для stdin")
    commands = parser.add_subparsers(dest="command", required=True, title="команды")
    ls = commands.add_parser("ls", help="имя, размер, дата изменения UTC и тип объектов")
    ls.add_argument("-a", "--all", action="store_true", help="показывать скрытые имена, '.' и '..'")
    ls.add_argument("path", nargs="?", default="/", metavar="ПУТЬ", help="абсолютный путь (по умолчанию '/')")
    cat = commands.add_parser("cat", help="байты файла в stdout без перекодирования")
    cat.add_argument("source", metavar="ФАЙЛ")
    cp = commands.add_parser("cp", help="сохранить файл с исходным именем в текущем каталоге")
    cp.add_argument("source", metavar="ФАЙЛ")
    tar = commands.add_parser("tar", help="сохранить каталог как ИМЯ.tar.gz в текущем каталоге")
    tar.add_argument("source", metavar="КАТАЛОГ")
    return parser


def safe_text(text: str) -> str:
    """Подготовить имя или сообщение для безопасного текстового вывода.

    Args:
        text: Строка, которая может содержать управляющие символы или суррогаты.

    Returns:
        Исходная печатная строка либо repr с экранированными символами.
    """
    return text if text.isprintable() else repr(text)


def format_time(nanoseconds: int) -> str:
    """Представить целочисленное время изменения в UTC.

    Args:
        nanoseconds: Знаковое число наносекунд от эпохи Unix.

    Returns:
        Дата с точной дробной частью либо число секунд вне диапазона datetime.
    """
    seconds, nanos = divmod(nanoseconds, 1_000_000_000)
    try:
        value = datetime(1970, 1, 1, tzinfo=timezone.utc) + timedelta(seconds=seconds)
        return value.strftime("%Y-%m-%d %H:%M:%S") + (f".{nanos:09d}" if nanos else "") + " UTC"
    except (ValueError, OverflowError):
        return f"{seconds} с от эпохи Unix"


def print_entry(fs: FileSystem, name: str, inode: Inode) -> None:
    """Вывести в stdout обязательные поля одного объекта.

    Args:
        fs: Источник цели, если inode является символьной ссылкой.
        name: Показываемое имя объекта.
        inode: Размер, время и тип объекта.

    Для ссылки дополнительно печатается цель; имена экранируются safe_text.
    """
    label = safe_text(name)
    if stat.S_ISLNK(inode.mode):
        label += " -> " + safe_text(fs.readlink(inode))
    print(f"{inode.size:>10} {format_time(inode.mtime_ns)} {inode.kind}: {label}")


def list_path(fs: FileSystem, path: str, show_all: bool) -> None:
    """Вывести один объект либо отсортированные записи каталога.

    Args:
        fs: Открытая файловая система.
        path: Абсолютный путь в образе.
        show_all: Включать скрытые имена, '.' и '..'.

    Последняя ссылка показывается как объект; порядок задают байты имён.
    """
    inode = fs.resolve(path, follow_final=False)
    if stat.S_ISDIR(inode.mode):
        entries = sorted(fs.directory(inode), key=lambda entry: entry.name.encode("utf-8", "surrogateescape"))
        for entry in entries:
            if show_all or not entry.name.startswith("."):
                print_entry(fs, entry.name, fs.inode(entry.inode))
    else:
        print_entry(fs, path, inode)


def dispatch(fs: FileSystem, args: argparse.Namespace) -> None:
    """Выполнить выбранную операцию над уже открытой файловой системой.

    Args:
        fs: Проверенный источник EXT4.
        args: Результат make_parser().parse_args с командой и исходным путём.

    Только cat пишет двоичные данные в stdout; cp и tar сообщают результат в stderr.
    """
    if args.command == "ls":
        list_path(fs, args.path, args.all)
    elif args.command == "cat":
        with fs.open_file(fs.resolve(args.source)) as input_stream:
            shutil.copyfileobj(input_stream, sys.stdout.buffer, length=IO_CHUNK_SIZE)
    elif args.command == "cp":
        target = copy_file(fs, args.source)
        print(f"Файл сохранён: {safe_text(str(target))}", file=sys.stderr)
    elif args.command == "tar":
        target = export_tar(fs, args.source)
        print(f"Архив сохранён: {safe_text(str(target))}", file=sys.stderr)


def main(argv: Sequence[str] | None = None) -> int:
    """Запустить CLI, преобразуя ожидаемые отказы в коды завершения.

    Args:
        argv: Аргументы без имени программы; None использует sys.argv.

    Returns:
        0 при успехе, 1 при ошибке операции или закрытом pipe, 130 при прерывании.

    Raises:
        SystemExit: Argparse завершил справку, вывод версии или ошибку синтаксиса.

    Программные ошибки, не относящиеся к ожидаемым отказам, не скрываются.
    """
    args = make_parser().parse_args(argv)
    try:
        with open_image(args.image, sys.stdin.buffer, args.offset) as image:
            fs = FileSystem(image)
            dispatch(fs, args)
        sys.stdout.flush()
        return 0
    except BrokenPipeError:
        # Подавляем повторный сброс закрытого stdout при завершении интерпретатора.
        try:
            descriptor = os.open(os.devnull, os.O_WRONLY)
            try:
                os.dup2(descriptor, sys.stdout.fileno())
            finally:
                os.close(descriptor)
        except OSError:
            pass
        return 1
    except (Ext4Error, OSError, ValueError, tarfile.TarError) as error:
        print(f"ext4py: ошибка: {safe_text(str(error))}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("ext4py: операция прервана", file=sys.stderr)
        return 130
