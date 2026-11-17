"""Выгрузка файла и каталога с атомарной публикацией готового результата.

Исходные пути относятся к EXT4, пути назначения — к хосту.
Существующие выходные имена, включая ссылки, не заменяются.
"""

from contextlib import contextmanager
import os
from pathlib import Path
import posixpath
import shutil
import stat
import tarfile
import tempfile
from typing import BinaryIO, Iterator

from .constants import IO_CHUNK_SIZE
from .errors import CorruptImageError, Ext4Error
from .filesystem import FileSystem
from .structures import Inode


def export_name(path: str) -> str:
    """Выбрать одно имя для файлового результата из пути EXT4.

    Args:
        path: Исходный путь, уже проверяемый файловым слоем.

    Returns:
        Последний компонент без завершающего '/', либо root для корня и точек.
    """
    name = posixpath.basename(path.rstrip("/"))
    return name if name and name not in (".", "..") else "root"


@contextmanager
def atomic_output(path: Path) -> Iterator[BinaryIO]:
    """Опубликовать завершённую запись без замены существующего имени.

    Args:
        path: Итоговый путь хоста с существующим родительским каталогом.

    Yields:
        Открытый двоичный поток временного файла рядом с назначением.

    При успешном выходе выполняются flush, fsync и os.link. При ошибке
    временное имя удаляется; существующее назначение не заменяется.
    Файловая система хоста должна поддерживать жёсткие ссылки.
    """
    if os.path.lexists(path):
        raise Ext4Error(f"Выходной файл уже существует: {str(path)!r}")
    fd, temporary = tempfile.mkstemp(prefix=".ext4py-", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as output:
            yield output
            output.flush()
            os.fsync(output.fileno())
        # link атомарно отказывает при занятом имени, в том числе при гонке.
        os.link(temporary, path)
    finally:
        os.unlink(temporary)


def copy_file(fs: FileSystem, source: str, destination: str | None = None) -> Path:
    """Выгрузить содержимое обычного файла, следуя по исходным ссылкам.

    Args:
        fs: Открытая файловая система EXT4.
        source: Абсолютный путь объекта в образе.
        destination: Путь хоста для внутреннего API; None означает исходное
            имя в cwd. Существующий каталог получает это имя внутри себя.

    Returns:
        Путь готовой копии на хосте.

    Права и владелец копии задаются хостом; байты записываются без перекодирования.
    """
    inode = fs.resolve(source)
    target = Path(destination) if destination is not None else Path(export_name(source))
    if destination is not None and target.is_dir():
        target /= export_name(source)
    with fs.open_file(inode) as input_stream, atomic_output(target) as output:
        shutil.copyfileobj(input_stream, output, length=IO_CHUNK_SIZE)
    return target


def _tar_info(fs: FileSystem, inode: Inode, name: str) -> tarfile.TarInfo:
    """Сформировать заголовок PAX без создания объектов на хосте.

    Args:
        fs: Источник, используемый для чтения цели символьной ссылки.
        inode: Метаданные архивируемого объекта.
        name: Относительное имя объекта внутри архива.

    Returns:
        TarInfo с типом, размером, правами, UID/GID и временем.

    Raises:
        Ext4Error: Тип объекта, например сокет, не поддержан экспортом.
    """
    info = tarfile.TarInfo(name)
    info.mode = stat.S_IMODE(inode.mode)
    info.uid, info.gid = inode.uid, inode.gid
    seconds, nanos = divmod(inode.mtime_ns, 1_000_000_000)
    info.mtime = seconds
    # Строка PAX сохраняет наносекунды без округления через float.
    if nanos:
        sign = "-" if inode.mtime_ns < 0 else ""
        whole, fraction = divmod(abs(inode.mtime_ns), 1_000_000_000)
        info.pax_headers["mtime"] = f"{sign}{whole}.{fraction:09d}"
    if stat.S_ISREG(inode.mode):
        info.type, info.size = tarfile.REGTYPE, inode.size
    elif stat.S_ISDIR(inode.mode):
        info.type = tarfile.DIRTYPE
    elif stat.S_ISLNK(inode.mode):
        info.type, info.linkname = tarfile.SYMTYPE, fs.readlink(inode)
    elif stat.S_ISFIFO(inode.mode):
        info.type = tarfile.FIFOTYPE
    elif stat.S_ISCHR(inode.mode) or stat.S_ISBLK(inode.mode):
        info.type = tarfile.CHRTYPE if stat.S_ISCHR(inode.mode) else tarfile.BLKTYPE
        old = int.from_bytes(inode.block_data[:4], "little")
        new = int.from_bytes(inode.block_data[4:8], "little")
        if old:
            info.devmajor, info.devminor = (old >> 8) & 255, old & 255
        else:
            info.devmajor = (new >> 8) & 0xFFF
            info.devminor = (new & 255) | ((new >> 12) & 0xFFF00)
    else:
        raise Ext4Error(f"Формат tar не поддерживает объект «{inode.kind}»: {name!r}")
    return info


def write_tar(fs: FileSystem, source: str, output: BinaryIO) -> None:
    """Записать каталог в поток tar.gz с сохранением дерева и ссылок.

    Args:
        fs: Открытая файловая система EXT4.
        source: Абсолютный путь каталога, корневая ссылка разыменовывается.
        output: Двоичный поток результата; закрывает вызывающая сторона.

    Дочерние символьные ссылки не обходятся. Жёсткие ссылки на обычные
    файлы повторно не копируют данные. Циклы каталогов и дубли имён
    вызывают ошибку; уже переданные в output байты откатить нельзя.
    """
    root = fs.resolve(source)
    if not stat.S_ISDIR(root.mode):
        raise Ext4Error("Для экспорта tar.gz требуется путь к каталогу")
    stack = [(root, export_name(source), frozenset())]
    hardlinks: dict[int, str] = {}
    with tarfile.open(fileobj=output, mode="w|gz", format=tarfile.PAX_FORMAT,
                      encoding="utf-8", errors="surrogateescape") as archive:
        while stack:
            inode, name, ancestors = stack.pop()
            info = _tar_info(fs, inode, name)
            if stat.S_ISDIR(inode.mode):
                if inode.number in ancestors:
                    raise CorruptImageError(f"Цикл каталогов: {name!r}")
                entries = [entry for entry in fs.directory(inode) if entry.name not in (".", "..")]
                if len(entries) != len({entry.name for entry in entries}):
                    raise CorruptImageError("Повторяющееся имя в архивируемом каталоге")
                archive.addfile(info)
                for entry in sorted(entries, key=lambda item: item.name.encode("utf-8", "surrogateescape"), reverse=True):
                    stack.append((fs.inode(entry.inode), f"{name}/{entry.name}", ancestors | {inode.number}))
            elif stat.S_ISREG(inode.mode):
                if inode.number in hardlinks:
                    info.type, info.linkname, info.size = tarfile.LNKTYPE, hardlinks[inode.number], 0
                    archive.addfile(info)
                else:
                    with fs.open_file(inode) as input_stream:
                        archive.addfile(info, input_stream)
                    hardlinks[inode.number] = name
            else:
                archive.addfile(info)


def export_tar(fs: FileSystem, source: str, destination: str | None = None) -> Path:
    """Сохранить каталог в завершённый архив tar.gz на хосте.

    Args:
        fs: Источник данных и метаданных EXT4.
        source: Абсолютный путь каталога в образе.
        destination: Итоговый путь хоста либо None для ИМЯ.tar.gz в cwd.

    Returns:
        Путь успешно опубликованного архива.

    При отказе запись не публикуется; существующее имя не перезаписывается.
    """
    target = Path(destination) if destination is not None else Path(export_name(source) + ".tar.gz")
    with atomic_output(target) as output:
        write_tar(fs, source, output)
    return target
