import matplotlib.pyplot as plt
from pathlib import Path
from typing import Iterable


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
