from abc import ABC, abstractmethod
from typing import Any


class BaseModel(ABC):
    """Абстрактный класс обёртки модели для получения
    предсказанного класса."""

    def __init__(self, model: Any):
        self.model = model

    @abstractmethod
    def predict(self, img: Any) -> float:
        """Предсказание вероятности положительного класса."""
        pass


class BaseTransform(ABC):
    """Абстрактный класс для преобразования изображения."""

    def __init__(self):
        pass

    @abstractmethod
    def transform(self, img: Any) -> Any:
        """Преобразование изображения.
        Могут быть дополнительные аргументы.
        Возвращает преобразованное изображение."""
        pass

    @abstractmethod
    @staticmethod
    def get_ranges() -> dict[str, list]:
        """Диапазон допустимых аргументов в формате словаря.
        Пример возвращающего словаря:
        {
            'scale_range': [0, float('inf')], <- int range
            'slant_range': [-180., 180.], <- float range
            'mode': ['blackbody', 'cied'], <- str list
        }

        Если число, то диапазон [from, to].
        Если str, то список допустимых категорий.
        """
        pass

