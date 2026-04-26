from abc import ABC, abstractmethod
from typing import Any
from utils.utils import ArgRange


class BaseModel(ABC):
    """Абстрактный класс обёртки модели для получения
    предсказанного класса"""

    def __init__(self, model: Any):
        self.model = model

    @abstractmethod
    def predict(self, img: Any) -> float:
        """Предсказание вероятности положительного класса"""
        pass


class BaseTransform(ABC):
    """Абстрактный класс для преобразования изображения"""

    def __init__(self):
        pass

    @abstractmethod
    def transform(self, img: Any) -> Any:
        """Преобразование изображения
        Могут быть дополнительные аргументы

        :param img: входное изображения для преобразования
        
        :return: преобразованное изображение"""
        pass

    @abstractmethod
    @staticmethod
    def get_ranges() -> dict[str, ArgRange]:
        """Возвращает диапазон допустимых аргументов.
        
        :return: допустимые диапазоны аргументов для
        данного трансформатора"""
        pass
