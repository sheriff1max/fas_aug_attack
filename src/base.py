from abc import ABC, abstractmethod
from typing import Any, Type, Literal
from .utils.dataclasses import ArgRange
from .utils.logging import LoggerOptuna
import optuna


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

    @staticmethod
    @abstractmethod
    def get_ranges() -> dict[str, ArgRange]:
        """Возвращает диапазон допустимых аргументов.
        
        :return: допустимые диапазоны аргументов для
        данного трансформатора"""
        pass


class BasePipelineAttackOptuna(ABC):
    """Абстрактный класс для нахождения лучших
    атак на модели с помощью Optuna.
    
    :param model: модель для атаки
    :param list_type_transforms: список типов-трансформаторов изображений
    :param logger: объект для логгирование экспериментов
    """

    def __init__(
        self,
        model: BaseModel,
        list_type_transforms: list[Type[BaseTransform]],
        logger: LoggerOptuna = None,
    ):
        self.model = model
        self.list_type_transforms = list_type_transforms
        self.logger = logger

        # Временное хранилище data, сохранённое в self.optimize(...)
        self._data = None

    @abstractmethod
    def _objective(self, trial: optuna.Trial) -> float:
        """Целевая функция для оптимизации

        :param trial: optuna для оптимизации
        :return: уверенность модели
        """

    def optimize(
        self, 
        data: Any,
        direction: Literal['minimize', 'maximize'] = 'minimize',
        n_trials: int = 100, 
        timeout: int = None,
        show_progress: bool = True,
        catch: tuple[type[Exception]] | type[Exception] = (),
    ) -> optuna.study.Study:
        """Запускает оптимизацию гиперпараметров трансформаций

        :param data: данные для атаки
        :param direction: направление оптимизации
        :param n_trials: количество итераций оптимизации
        :param timeout: лимит времени в секундах
        :param show_progress: показывать прогресс оптимизации или нет
        :param catch: какие ошибки отлавливать при оптимизации (ValueError и т.д.)
        :return: результаты оптимизации
        """
        if self.logger:
            self.logger.start()

        # Сохраняем данные для доступа из _objective
        self._data = data

        study = optuna.create_study(
            direction=direction,
        )

        study.optimize(
            self._objective, 
            n_trials=n_trials, 
            timeout=timeout,
            show_progress_bar=show_progress,
            catch=catch,
        )

        if self.logger:
            dict_importance = optuna.importance.get_param_importances(study)
            self.logger.end(dict_importance=dict_importance)

        self._data = None
        return study
