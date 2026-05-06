from enum import Enum


class DataType(Enum):
    """Enum для типа данных допустимного
    диапазона аргументов."""
    INT = 1
    FLOAT = 2
    STR = 3
    BOOL = 4


class Inf(Enum):
    """Enum для ограничения `бесконечных` диапазонов
    подбираемых аргументов."""
    TINY = 3
    SMALL = 10
    MEDIUM = 25
    BIG = 50
    LARGE = 100
