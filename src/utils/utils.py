import optuna

from typing import Any, Type

from .enums import DataType
from ..base import BaseTransform


def make_list_transforms_optuna(
    trial: optuna.Trial,
    list_type_transforms: list[Type[BaseTransform]],
    optimize_set_transforms: bool = False,
) -> list[BaseTransform]:
    """Функция создаёт объекты преобразований с оптимальными
    параметрами, оптимизированных с помощью Optuna

    :param trial:
    :param list_type_transforms: список типов трансформации (не объектов)
    :param optimize_set_transforms: оптимизировать ли набор трансформаторов

    :return: список преобразований с оптимальными аргументами
    """
    list_transforms = []

    for type_transform in list_type_transforms:
        params = get_ranges2optuna(
            trial=trial,
            type_transform=type_transform,
            optimize_set_transforms=optimize_set_transforms
        )
        if params:
            transform_instance = type_transform(**params)
            list_transforms.append(transform_instance)
    return list_transforms


def get_ranges2optuna(
    trial: optuna.Trial,
    type_transform: Type[BaseTransform],
    optimize_set_transforms: bool = False,
) -> dict[str, Any] | None:
    """Функция для модификации формата границ
    преобразователя BaseTransform в формат optuna
    для дальнейшей оптимизации.

    :param trial:
    :param type_transform: тип трансформации (не объект).
    :param optimize_set_transforms: оптимизировать ли набор трансформаторов

    :return: подобранные аргументы для type_transform.
    """
    class_name = type_transform.__name__
    ranges = type_transform.get_ranges()

    if optimize_set_transforms:
        flag_include = trial.suggest_categorical(
            f'flag_include--{class_name}',
            [True, False]
        )
        if not flag_include:
            return

    new_ranges = dict()
    for key, arg_range in ranges.items():
        if arg_range.is_tuple:
            
            if arg_range.data_type == DataType.INT:
                left = trial.suggest_int(
                    f'{key}--left--{class_name}',
                    arg_range.values[0],
                    arg_range.values[1] - 2,
                )
                right = trial.suggest_int(
                    f'{key}--right--{class_name}',
                    left + 1,
                    arg_range.values[1],
                )
            elif arg_range.data_type == DataType.FLOAT:
                left = trial.suggest_float(
                    f'{key}--left--{class_name}',
                    arg_range.values[0],
                    arg_range.values[1] - 2e-6,
                )
                right = trial.suggest_float(
                    f'{key}--right--{class_name}',
                    left + 1e-6,
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
