from base import BaseModel, BaseTransform
from typing import Any, Type, Literal
import optuna
from utils.utils import get_ranges2optuna


class AttackPipeline:
    """Pipeline-класс, внутри которого хранятся модель
    и преобразования для изображений."""

    def __init__(
        self,
        model: BaseModel,
        list_transforms: list[BaseTransform],
    ):
        self.model = model
        self.list_transforms = list_transforms

    def attack(self, img: Any) -> float:
        """Предсказание вероятности положительного класса при
        применённых преобразованиях данных.

        :param img: Any - изображение формата, с которым работает
        BaseTransform и BaseModel.
        :return: float - вероятность положительного класса.
        """
        for transform in self.list_transforms:
            img = transform.transform(img)
        return self.model.predict(img)


class OptunaAttackPipeline:
    """Pipeline-класс для поиска наилучших преобразований
    входных данных, которые наилучшим образом обманывают модель"""

    def __init__(
        self,
        model: BaseModel,
        list_type_transforms: list[Type[BaseTransform]],
    ):
        self.model = model
        self.list_type_transforms = list_type_transforms
        # Временное хранилище для изображения в процессе оптимизации
        self._current_img = None

    def _objective(self, trial: optuna.Trial) -> float:
        """
        Целевая функция для оптимизации
        :param trial: optuna для оптимизации
        :return: вероятность положительного класса
        """
        list_transforms = []

        for type_transform in self.list_type_transforms:
            params = get_ranges2optuna(trial, type_transform)
            transform_instance = type_transform(**params)
            list_transforms.append(transform_instance)

        attack_pipeline = AttackPipeline(
            model=self.model,
            list_transforms=list_transforms,
        )

        score = attack_pipeline.attack(self._current_img)
        return score

    def optimize(
        self, 
        img: Any,
        direction: Literal['minimize', 'maximize'] = 'minimize',
        study_name: str = 'attack_optimization',
        n_trials: int = 100, 
        timeout: int = None,
        show_progress: bool = True,
    ) -> optuna.study.Study:
        """
        Запускает оптимизацию гиперпараметров трансформаций

        :param img: изображение для атаки
        :param n_trials: количество итераций оптимизации
        :param timeout: лимит времени в секундах
        :return: метаданные оптимизации
        """
        # Сохраняем изображение для доступа из _objective
        self._current_img = img

        study = optuna.create_study(
            direction=direction,
            study_name=study_name,
        )

        study.optimize(
            self._objective, 
            n_trials=n_trials, 
            timeout=timeout,
            show_progress_bar=show_progress
        )

        # Очищаем временное изображение
        self._current_img = None

        return study
