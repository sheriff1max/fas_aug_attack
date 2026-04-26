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
    TINY = 25
    SMALL = 100
    MEDIUM = 250
    BIG = 500
    LARGE = 1000
