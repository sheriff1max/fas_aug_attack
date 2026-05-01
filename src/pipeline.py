from .base import BaseModel, BaseTransform
from typing import Any, Type, Literal
import optuna
from .utils.utils import get_ranges2optuna
from .utils.logging import LoggerOptuna
from .utils.dataclasses import ResponsePipelineAttackImg
from optuna.importance import get_param_importances


optuna.logging.set_verbosity(optuna.logging.WARNING)


class PipelineAttackImg:
    """Pipeline-класс, внутри которого хранятся модель
    и преобразования для изображений."""

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


class OptunaPipelineAttackImgOptuna:
    """Pipeline-класс для поиска наилучших преобразований
    входных данных, которые наилучшим образом обманывают модель"""

    def __init__(
        self,
        model: BaseModel,
        list_type_transforms: list[Type[BaseTransform]],
        logger: LoggerOptuna = None,
    ):
        self.model = model
        self.list_type_transforms = list_type_transforms
        self.logger = logger
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

        attack_pipeline = PipelineAttackImg(
            model=self.model,
            list_transforms=list_transforms,
        )

        response = attack_pipeline.attack(self._current_img)
        score = response.score

        if self.logger:
            self.logger.step(
                img=response.img,
                score=score,
                step=trial.number,
                params=trial.params,
            )
        return score

    def optimize(
        self, 
        img: Any,
        direction: Literal['minimize', 'maximize'] = 'minimize',
        n_trials: int = 100, 
        timeout: int = None,
        show_progress: bool = True,
        catch: tuple[type[Exception]] | type[Exception] = (),
    ) -> optuna.study.Study:
        """
        Запускает оптимизацию гиперпараметров трансформаций

        :param img: изображение для атаки
        :param direction: направление оптимизации
        :param n_trials: количество итераций оптимизации
        :param timeout: лимит времени в секундах
        :param show_progress: показывать прогресс оптимизации или нет
        :param catch: какие ошибки отлавливать при оптимизации (ValueError и т.д.)

        :return: метаданные оптимизации
        """
        if self.logger:
            self.logger.start()

        # Сохраняем изображение для доступа из _objective
        self._current_img = img

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
            self.logger.end(dict_importance=get_param_importances(study))

        # Очищаем временное изображение
        self._current_img = None

        return study


class OptunaPipelineAttackDatasetOptuna:
    """"""
