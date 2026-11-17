"""Ограниченное чтение raw-образа и подготовка stdin для произвольного доступа.

Смещения Image отсчитываются от начала EXT4, а не от начала файла хоста.
Входной файл никогда не открывается для записи.
"""

from contextlib import contextmanager
import shutil
import tempfile
from typing import BinaryIO, Iterator

from .constants import IO_CHUNK_SIZE
from .errors import CorruptImageError, Ext4Error


class Image:
    """Представление EXT4 внутри двоичного потока со смещением.

    Attributes:
        stream: Заимствованный поток с seek; закрывает его вызывающая сторона.
        offset: Начало EXT4 в файле хоста, в байтах.
        size: Доступная длина после offset, в байтах.
    """

    def __init__(self, stream: BinaryIO, offset: int = 0):
        """Определить длину представления и проверить смещение.

        Args:
            stream: Открытый двоичный поток с поддержкой seek и tell.
            offset: Неотрицательное начало EXT4 относительно файла хоста.

        Raises:
            Ext4Error: Смещение отрицательно.
            CorruptImageError: После смещения доступно меньше 2048 байт.
        """
        if offset < 0:
            raise Ext4Error("Смещение не может быть отрицательным")
        stream.seek(0, 2)
        self.size = stream.tell() - offset
        if self.size < 2048:
            raise CorruptImageError("Образ слишком мал или смещение выходит за его границы")
        self.stream = stream
        self.offset = offset

    def read_at(self, offset: int, size: int) -> bytes:
        """Прочитать ровно size байт относительно начала EXT4.

        Args:
            offset: Неотрицательный байтовый адрес внутри представления.
            size: Число запрашиваемых байт; нулевая длина допустима.

        Returns:
            Байты указанного диапазона.

        Raises:
            CorruptImageError: Диапазон выходит за границы или поток усечён.
            OSError: Чтение или позиционирование завершилось ошибкой хоста.
        """
        if offset < 0 or size < 0 or offset + size > self.size:
            raise CorruptImageError(f"Чтение за границами образа: смещение {offset}, размер {size}")
        self.stream.seek(self.offset + offset)
        data = self.stream.read(size)
        if len(data) != size:
            raise CorruptImageError("Образ усечён во время чтения")
        return data


@contextmanager
def open_image(path: str, stdin: BinaryIO, offset: int = 0) -> Iterator[Image]:
    """Открыть raw-файл либо сохранить stdin во временный файл.

    Args:
        path: Путь хоста или прочерк для чтения stdin.
        stdin: Двоичный источник для случая path == '-'.
        offset: Начало EXT4 в байтах входного образа.

    Yields:
        Image, доступный только внутри контекста with.

    Временный файл и открытый вход закрываются при выходе из контекста,
    включая исключения. Весь stdin принимается до начала разбора.
    """
    if path == "-":
        with tempfile.TemporaryFile(prefix="ext4py-") as stream:
            shutil.copyfileobj(stdin, stream, length=IO_CHUNK_SIZE)
            stream.flush()
            yield Image(stream, offset)
    else:
        with open(path, "rb") as stream:
            yield Image(stream, offset)
