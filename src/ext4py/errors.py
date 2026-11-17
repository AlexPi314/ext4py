"""Ожидаемые ошибки операций, повреждения образа и неподдерживаемого формата."""

class Ext4Error(Exception):
    """Ожидаемая ошибка операции, показываемая CLI без traceback."""

    pass


class CorruptImageError(Ext4Error):
    """Усечённые, повреждённые или несогласованные структуры входного образа."""

    pass


class UnsupportedFeatureError(Ext4Error):
    """Заведомо не реализованная возможность формата EXT4."""

    pass
