from dataclasses import dataclass
from typing import Any
from .enums import DataType


@dataclass
class ArgRange:
    """Класс для представления диапазона допустимых аргументов
    конкретного преобразующего метода.
    
    :param values: список состоящий либо из 2, либо из N элементов.
        Примеры:
        - [0, 15] -> диапазон от 0 до 15 включительно.
        - [0.0, 1.0] -> диапазон от 0.0 до 1.0 включительно.
        - ['category1', 'category2', 'category3'] -> доступные категории для перебора.
        - [True, False] -> доступные значения для перебора.
    :param data_type: тип данных у диапазона.
    :param is_tuple: аргумент состоит из одного числа или множества.
    """
    values: list[int | float | str]
    data_type: DataType
    is_tuple: bool = False


@dataclass
class ResponsePipelineAttackImg:
    """"""
    img: Any
    score: float
