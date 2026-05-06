import optuna
from typing import Any, Type

from .enums import DataType
from ..base import BaseTransform


def make_list_transforms_optuna(
    trial: optuna.Trial,
    list_type_transforms: list[list[Type[BaseTransform]]],
) -> list[BaseTransform]:
    """Функция создаёт объекты преобразований с оптимальными
    параметрами, оптимизированных с помощью Optuna

    :param trial:
    :param list_type_transforms: список списков типов трансформации (не объектов)
    :param optimize_set_transforms: оптимизировать ли набор трансформаторов

    :return: список преобразований с оптимальными аргументами
    """
    list_transforms = []

    for i, sublist_type_transforms in enumerate(list_type_transforms):
        if len(sublist_type_transforms) > 1:
            idx_choice = trial.suggest_int(
                f'idx_choice--{i}',
                0,
                len(sublist_type_transforms) - 1,
            )
        else:
            idx_choice = 0

        type_transform = sublist_type_transforms[idx_choice]
        params = get_ranges2optuna(
            trial=trial,
            type_transform=type_transform,
        )
        transform_instance = type_transform(**params)
        list_transforms.append(transform_instance)
    return list_transforms


def get_ranges2optuna(
    trial: optuna.Trial,
    type_transform: Type[BaseTransform],
) -> dict[str, Any] | None:
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
