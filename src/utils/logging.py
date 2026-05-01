import json
from pathlib import Path
from PIL import Image
import numpy as np
from typing import Any, Literal
from collections import defaultdict
import pandas as pd
from .utils import save_plot, save_importance_barh


class LoggerOptuna:
    """Класс для логирования оптимизации атак Optuna.
    Сохраняет:
        - метрики +
        - список оптимизируемых параметров +
        - название модели для атаки
        - список типов преобразований
        - преобразованные картинки +

    :param experiment_name:
    """
    FILENAME_METAINFO = 'METAINFO.json'
    FILENAME_BEST_PARAMS = 'best_params.json'
    FOLDER_NAME4IMGS = 'examples'

    def __init__(
        self,
        direction: Literal['minimize', 'maximize'],
        path: str = 'logs',
        experiment_name: str = 'run',
        description: str | None = None,
    ):
        self.path = Path(path)
        self.path.mkdir(parents=True, exist_ok=True)

        self.direction = direction
        self.experiment_name = experiment_name
        self.description = description

        self._flag_start = False

    def start(self) -> None:
        """Создаёт новую папку для сохранения примеров.

        :return: None
        """
        self._flag_start = True

        existing = [
            d for d in self.path.iterdir() \
                if d.is_dir() and \
                d.name.startswith(f"{self.experiment_name}_")
        ]
        next_idx = len(existing) + 1

        self._cur_run_path = self.path / f"{self.experiment_name}_{next_idx}"
        self._cur_run_path.mkdir(parents=True, exist_ok=True)

        self._cur_run_imgs_path = self._cur_run_path / self.FOLDER_NAME4IMGS
        self._cur_run_imgs_path.mkdir(parents=True, exist_ok=True)

        self._logs = defaultdict(list)
        self._best_score = None
        self._meta_saved = False

    def step(
        self,
        img: Any,
        score: float,
        step: int,
        params: dict,
    ) -> None:
        """Сравнивает score с лучшим результатом
        и сохраняет изображение при улучшении.

        :param img:
        :param score:
        :param step:
        :param params:
        :return:
        """
        self._check_start()

        is_better = False
        if self._best_score is None:
            is_better = True
        elif self.direction == "maximize" and score > self._best_score:
            is_better = True
        elif self.direction == "minimize" and score < self._best_score:
            is_better = True

        if is_better:
            self._best_score = score
            filename = f"step_{step}_score={score:.5f}.png"
            filepath = self._cur_run_imgs_path / filename
            self._save_image(img=img, path=filepath)

        self._logs['step'].append(step)
        self._logs['score'].append(score)
        self._logs['params'].append(params)
        self._save_metainfo(params=params)

    def end(self, dict_importance: dict[str, float]) -> None:
        """"""
        # График изменения метрики с шагами оптимизации.
        self._flag_start = False

        df_score = pd.DataFrame(self._logs).sort_values(by='step')
        df_score.to_csv(self._cur_run_path / 'scores.csv', index=False)
        save_plot(
            x=df_score['step'].values,
            y=df_score['score'].values,
            path=self._cur_run_path / 'scores_plot.png',
            title='График изменения score во время оптимизации',
            xlabel='Шаг',
            ylabel='score',
        )

        # Сохранение лучших найденных параметров для атаки на модель.
        best_row: pd.DataFrame = df_score.sort_values(by='score', ignore_index=True)
        best_params = best_row.loc[0, 'params']
        best_score = best_row.loc[0, 'score']
        data = {'best_params': best_params, 'best_score': best_score}

        path2best_params = self._cur_run_path / self.FILENAME_BEST_PARAMS
        with open(path2best_params, "w") as f:
            json.dump(data, f)

        # График важности параметров.
        df_importance = pd.DataFrame(
            {
                'param': list(dict_importance.keys()),
                'value': list(dict_importance.values()),
            }
        )
        df_importance = df_importance.sort_values(by='value', ascending=False)
        df_importance.to_csv(self._cur_run_path / 'param_importance.csv', index=False)

        df_importance = df_importance.head(15)
        save_importance_barh(
            names=df_importance['param'].values,
            values=df_importance['value'].values,
            path=self._cur_run_path / 'param_importance_plot.png',
            title='Важность параметров на атаки',
            xlabel='Важность',
            ylabel='Параметр',
        )

    def _save_image(
        self,
        img: Any,
        path: Path
    ) -> None:
        """Сохранение изображения

        :param img:
        :param Path:
        :return:
        """
        if not isinstance(img, np.ndarray):
            img = np.array(img)

        if img.max() <= 1.0:
            img = (img * 255).astype(np.uint8)
        elif img.dtype != np.uint8:
            img = img.astype(np.uint8)
        Image.fromarray(img).save(path)

    def _save_metainfo(self, params: dict) -> None:
        """Сохранение метаинформации об оптимизации

        :param params:
        :return:
        """
        if not self._meta_saved:
            list_transforms = list(set([
                param.split('--')[-1] for param in params
            ]))
            data = {
                'list_transforms': list_transforms,
                'description': self.description,
            }

            path2meta = self._cur_run_path / self.FILENAME_METAINFO
            with open(path2meta, "w") as f:
                json.dump(data, f)

            self._meta_saved = True

    def _check_start(self) -> None:
        """"""
        if not self._check_start:
            raise Exception('You should call .start() firstly.')
        
