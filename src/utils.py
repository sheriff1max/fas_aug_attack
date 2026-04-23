# TODO: сделать преобразующую
# функцию BaseTransform.get_ranges() в optuna перебор


import optuna


def get_ranges2optuna(
    trial: optuna.Trial,
    class_name: str,
    ranges: dict[str, list],
) -> dict[str, list]:
    """"""
    new_ranges = dict()
    for name, lst in ranges.items():
        val = lst[0]
        if isinstance(val, float):
            pass
        elif isinstance(val, int):
            pass
        else:
            pass
    return