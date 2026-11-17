"""Отображение логических блоков файла в физические диапазоны EXT4.

Поддерживаются extent-деревья и прямые/косвенные указатели.
Нулевые области представлены физическим адресом None.
"""

import struct
from typing import Callable, Iterator

from .constants import (
    EXTENT_MAGIC,
    INODE_EXTENTS,
    MAX_EXTENT_DEPTH,
    RO_COMPAT_METADATA_CSUM,
    UNWRITTEN_EXTENT_BASE,
)
from .errors import CorruptImageError
from .structures import Inode, Superblock, crc32c, u32

BlockRange = tuple[int, int, int | None]


class BlockMap:
    """Читатель карты блоков одного inode.

    Диапазон имеет вид (logical, count, physical): адреса заданы в блоках,
    physical=None означает нули. Пропуски между extent дополняет FileSystem.
    """

    def __init__(self, sb: Superblock, read_block: Callable[[int], bytes], inode: Inode):
        """Связать inode с геометрией и функцией чтения блока.

        Args:
            sb: Проверенные параметры файловой системы.
            read_block: Функция физического чтения одного блока по его номеру.
            inode: Объект, карту которого требуется обойти.
        """
        self.sb = sb
        self.read_block = read_block
        self.inode = inode
        self.limit = (inode.size + sb.block_size - 1) // sb.block_size
        self.seed = crc32c(struct.pack("<II", inode.number, inode.generation), sb.checksum_seed)

    def _check_physical(self, block: int, count: int = 1) -> None:
        """Проверить выделенный диапазон физических блоков.

        Args:
            block: Номер первого блока; ноль не обозначает выделенные данные.
            count: Положительное число последовательных блоков.

        Raises:
            CorruptImageError: Диапазон пуст, отрицателен или выходит за EXT4.
        """
        if block <= 0 or count <= 0 or block + count > self.sb.blocks_count:
            raise CorruptImageError(f"Некорректный адрес блока в inode {self.inode.number}")

    def ranges(self) -> Iterator[BlockRange]:
        """Выдать карту файла по extent или старой адресации.

        Yields:
            BlockRange в порядке логических адресов. Для старой адресации
            обход ограничен логическим размером файла; extent могут быть за EOF.

        Проверки выполняются лениво во время обхода итератора.
        """
        if self.inode.flags & INODE_EXTENTS:
            yield from self._extents(self.inode.block_data, None, 0, 1 << 32, set(), True)
        else:
            pointers = struct.unpack("<15I", self.inode.block_data)
            capacity = 12 + sum((self.sb.block_size // 4) ** depth for depth in (1, 2, 3))
            if self.limit > capacity:
                raise CorruptImageError("Размер файла превышает ёмкость косвенной адресации")
            for logical, pointer in enumerate(pointers[:12]):
                if logical >= self.limit:
                    return
                if pointer:
                    self._check_physical(pointer)
                yield logical, 1, pointer or None
            logical = 12
            for depth, pointer in enumerate(pointers[12:], 1):
                if logical >= self.limit:
                    return
                yield from self._indirect(pointer, depth, logical, set())
                logical += (self.sb.block_size // 4) ** depth

    def _extents(self, data: bytes, expected_depth: int | None, lower: int, upper: int,
                 ancestors: set[int], root: bool = False) -> Iterator[BlockRange]:
        """Проверить узел extent-дерева и выдать его листовые диапазоны.

        Args:
            data: Корень из i_block либо полный внешний блок узла.
            expected_depth: Глубина, ожидаемая родителем; None для корня.
            lower: Включённая нижняя граница логических блоков поддерева.
            upper: Исключённая верхняя граница логических блоков поддерева.
            ancestors: Физические адреса узлов текущей ветви для поиска цикла.
            root: Признак корня, защищаемого CRC самого inode.

        Yields:
            Проверенные листовые диапазоны, включая unwritten как None.

        Raises:
            CorruptImageError: Нарушены заголовок, глубина, CRC или границы.
        """
        magic, entries, maximum, depth = struct.unpack_from("<4H", data)
        if (magic != EXTENT_MAGIC or depth > MAX_EXTENT_DEPTH or not maximum
                or maximum > (len(data) - 12) // 12 or entries > maximum
                or (expected_depth is not None and depth != expected_depth)
                or (depth and not entries)):
            raise CorruptImageError("Повреждён заголовок extent-дерева")
        if not root and self.sb.ro_compat & RO_COMPAT_METADATA_CSUM:
            tail = 12 + maximum * 12
            if tail + 4 > len(data) or crc32c(data[:tail], self.seed) != u32(data, tail):
                raise CorruptImageError("Не совпадает контрольная сумма узла extent-дерева")
        previous_end = lower
        keys = [u32(data, 12 + i * 12) for i in range(entries)]
        if any(a >= b for a, b in zip(keys, keys[1:])):
            raise CorruptImageError("Нарушен порядок записей extent-дерева")
        for index, logical in enumerate(keys):
            if logical < previous_end or logical >= upper:
                raise CorruptImageError("Перекрытие диапазонов extent-дерева")
            pos = 12 + index * 12
            if depth:
                physical = u32(data, pos + 4) | (struct.unpack_from("<H", data, pos + 8)[0] << 32)
                self._check_physical(physical)
                if physical in ancestors:
                    raise CorruptImageError("Цикл в extent-дереве")
                end = keys[index + 1] if index + 1 < entries else upper
                yield from self._extents(self.read_block(physical), depth - 1,
                                         logical, end, ancestors | {physical})
                previous_end = end
            else:
                length, hi, lo = struct.unpack_from("<HHI", data, pos + 4)
                # 32768 ещё означает обычный extent; unwritten начинается с 32769.
                unwritten = length > UNWRITTEN_EXTENT_BASE
                count = length - UNWRITTEN_EXTENT_BASE if unwritten else length
                if not count or logical + count > upper:
                    raise CorruptImageError("Некорректная длина extent")
                physical = lo | (hi << 32)
                self._check_physical(physical, count)
                yield logical, count, None if unwritten else physical
                previous_end = logical + count

    def _indirect(self, pointer: int, depth: int, logical: int,
                  ancestors: set[int]) -> Iterator[BlockRange]:
        """Обойти косвенное поддерево, не разворачивая пустые ветви.

        Args:
            pointer: Адрес косвенного блока или ноль для пустого поддерева.
            depth: Число оставшихся уровней косвенности, от 1 до 3.
            logical: Первый логический блок поддерева.
            ancestors: Адреса предков для проверки циклов.

        Yields:
            Диапазоны данных или нулей до конца логического файла.
        """
        per_block = self.sb.block_size // 4
        capacity = per_block ** depth
        # Нулевое поддерево представляется одним диапазоном без чтения его блоков.
        if not pointer:
            yield logical, min(capacity, self.limit - logical), None
            return
        self._check_physical(pointer)
        if pointer in ancestors:
            raise CorruptImageError("Цикл в косвенных блоках")
        data = self.read_block(pointer)
        span = per_block ** (depth - 1)
        for index, (child,) in enumerate(struct.iter_unpack("<I", data)):
            start = logical + index * span
            if start >= self.limit:
                return
            if depth == 1:
                if child:
                    self._check_physical(child)
                yield start, 1, child or None
            else:
                yield from self._indirect(child, depth - 1, start, ancestors | {pointer})
