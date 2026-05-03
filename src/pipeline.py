from .base import BaseModel, BaseTransform, BasePipelineAttackOptuna
from .utils.utils import get_ranges2optuna
from .utils.logging import LoggerOptuna
from .utils.dataclasses import ResponsePipelineAttackImg

from typing import Any, Type
import optuna
import numpy as np


optuna.logging.set_verbosity(optuna.logging.WARNING)


class PipelineAttackImg:
    """Pipeline-класс, внутри которого хранятся модель
    и преобразования для изображений.
    
    :param model: модель для атаки
    :param list_transforms: список объектов-трансформаторов изображений
    """

    def __init__(
        self,
        model: BaseModel,
        list_transforms: list[BaseTransform],
    ):
        self.model = model
        self.list_transforms = list_transforms

    def attack(self, img: Any) -> ResponsePipelineAttackImg:
        """Предсказание вероятности положительного класса при
        применённых преобразованиях данных.

        :param img: Any - изображение формата, с которым работает
        BaseTransform и BaseModel.
        :return: словарь результирующих метаданных.
        """
        for transform in self.list_transforms:
            img = transform.transform(img)
        return ResponsePipelineAttackImg(
            img=img,
            score=self.model.predict(img)
        )


class PipelineAttackOptunaImg(BasePipelineAttackOptuna):
    """Pipeline-класс для поиска наилучших преобразований
    для одного изображения, которые наилучшим образом обманывают модель"""

    def __init__(
        self,
        model: BaseModel,
        list_type_transforms: list[Type[BaseTransform]],
        logger: LoggerOptuna = None,
    ):
        super().__init__(
            model=model,
            list_type_transforms=list_type_transforms,
            logger=logger,
        )
        self._data = None

    def _objective(self, trial: optuna.Trial) -> float:
        """Целевая функция для оптимизации

        :param trial: optuna для оптимизации
        :return: уверенность модели
        """
        list_transforms = []

        for type_transform in self.list_type_transforms:
            params = get_ranges2optuna(trial, type_transform)
            transform_instance = type_transform(**params)
            list_transforms.append(transform_instance)

        attack_pipeline = PipelineAttackImg(
            model=self.model,
            list_transforms=list_transforms,
        )

        response = attack_pipeline.attack(self._data)
        score = response.score

        if self.logger:
            self.logger.step(
                img=response.img,
                score=score,
                step=trial.number,
                params=trial.params,
            )
        return score


class PipelineAttackOptunaDataset(BasePipelineAttackOptuna):
    """Pipeline-класс для поиска наилучших преобразований,
    которые наилучшим образом обманывают модель на всём
    датасете в среднем
    """

    def __init__(
        self,
        model: BaseModel,
        list_type_transforms: list[Type[BaseTransform]],
        logger: LoggerOptuna = None,
    ):
        super().__init__(
            model=model,
            list_type_transforms=list_type_transforms,
            logger=logger,
        )
        self._data = None

    def _objective(self, trial: optuna.Trial) -> float:
        """Целевая функция для оптимизации

        :param trial: optuna для оптимизации
        :return: уверенность модели
        """
        list_transforms = []

        for type_transform in self.list_type_transforms:
            params = get_ranges2optuna(trial, type_transform)
            transform_instance = type_transform(**params)
            list_transforms.append(transform_instance)

        attack_pipeline = PipelineAttackImg(
            model=self.model,
            list_transforms=list_transforms,
        )

        list_scores = []
        for i in range(len(self._data)):
            img = self._data[i]['img']
            response = attack_pipeline.attack(img)
            score = response.score

            list_scores.append(score)

        score = np.mean(list_scores)

        if self.logger:
            self.logger.step(
                img=response.img,
                score=score,
                step=trial.number,
                params=trial.params,
            )
        return score
