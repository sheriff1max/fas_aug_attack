import optuna
import matplotlib.pyplot as plt

from typing import Any, Type, Iterable
from pathlib import Path

from .enums import DataType
from ..base import BaseTransform


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


def save_plot(
    x: Iterable,
    y: Iterable,
    path: Path,
    figsize: tuple = (10, 5),
    title: str | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
) -> None:
    """"""
    plt.figure(figsize=figsize)
    plt.plot(x, y)

    if title:
        plt.title(title)
    if xlabel:
        plt.xlabel(xlabel)
    if ylabel:
        plt.ylabel(ylabel)

    plt.grid(True)
    plt.savefig(path)
    plt.close()


def save_importance_barh(
    names: Iterable,
    values: Iterable,
    path: Path,
    figsize: tuple = (10, 5),
    title: str | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    color: str = '#2196F3',
) -> None:
    """"""
    plt.figure(figsize=figsize)
    bars = plt.barh(
        names,
        values,
        color=color,
    )

    for bar, val in zip(bars, values):
        width = bar.get_width()
        plt.text(width + max(values) * 0.01, bar.get_y() + bar.get_height() / 2, 
                f'{val:.4f}', ha='left', va='center', fontsize=8)

    if xlabel:
        plt.xlabel(xlabel)
    if ylabel:
        plt.ylabel(ylabel)
    if title:
        plt.title(title)

    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
