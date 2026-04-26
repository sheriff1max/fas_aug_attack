import optuna
from utils.enums import DataType
from typing import Any, Type
from base import BaseTransform
from dataclasses import dataclass


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


def get_ranges2optuna(
    trial: optuna.Trial,
    type_transform: Type[BaseTransform],
) -> dict[str, Any]:
    """Функция для модификации формата границ
    преобразователя BaseTransform в формат optuna
    для дальнейшей оптимизации.
    
    :param trial:
    :param type_transform: тип трансформации (не объект).
    
    :return: подобранные аргументы для type_transform.
    """

    class_name = type_transform.__name__
    ranges = type_transform.get_ranges()

    new_ranges = dict()
    for key, arg_range in ranges.items():
        if arg_range.is_tuple:
            
            if arg_range.data_type == DataType.INT:
                left = trial.suggest_int(
                    f'{key}--left--{class_name}',
                    arg_range.values[0],
                    arg_range.values[1],
                )
                right = trial.suggest_int(
                    f'{key}--right--{class_name}',
                    left,
                    arg_range.values[1],
                )
            elif arg_range.data_type == DataType.FLOAT:
                left = trial.suggest_float(
                    f'{key}--left--{class_name}',
                    arg_range.values[0],
                    arg_range.values[1],
                )
                right = trial.suggest_float(
                    f'{key}--right--{class_name}',
                    left,
                    arg_range.values[1],
                )
            else:
                raise ValueError(f'key `{key}` has data_type = `{arg_range.data_type}`')
            new_ranges[key] = (left, right)
        
        else:
            if arg_range.data_type == DataType.INT:
                value = trial.suggest_int(
                    f'{key}--{class_name}',
                    arg_range.values[0],
                    arg_range.values[1],
                )
            elif arg_range.data_type == DataType.FLOAT:
                value = trial.suggest_float(
                    f'{key}--{class_name}',
                    arg_range.values[0],
                    arg_range.values[1],
                )
            elif arg_range.data_type in (DataType.BOOL, DataType.STR):
                value = trial.suggest_categorical(
                    f'{key}--{class_name}',
                    arg_range.values,
                )
            else:
                raise ValueError(f'key `{key}` has data_type = `{arg_range.data_type}`')
            
            new_ranges[key] = value

    return new_ranges
