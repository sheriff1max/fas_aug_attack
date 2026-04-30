import os
import mlflow
import optuna
from pathlib import Path
from PIL import Image
import numpy as np
from typing import Any, Literal


class LoggerOptuna:
    """Класс для логирования оптимизации атак Optuna с mlflow"""

    def __init__(
        self,
        folder_name: str,
        experiment_name: str,
        direction: Literal['minimize', 'maximize'],
        folder_name_examples: str = 'example',
        tracking_uri_name: str = 'mlflow',
    ):
        self.folder_name = Path(folder_name)
        self.folder_name.mkdir(parents=True, exist_ok=True)

        self.folder_name_examples = folder_name_examples
        self.direction = direction

        tracking_uri = self.folder_name / tracking_uri_name
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment_name)

        self.current_example_dir = None
        self.best_score = None

        self.flag_start = False

    def start(self) -> None:
        """Инициализирует папку эксперимента и
        MLflow run для нового изображения.

        :return: None
        """
        self.flag_start = True

        existing = [d for d in self.folder_name.iterdir() if d.is_dir() and d.name.startswith(f"{self.folder_name_examples}_")]
        next_idx = len(existing) + 1

        self.current_example_dir = self.folder_name / f"{self.folder_name_examples}_{next_idx}"
        self.current_example_dir.mkdir(parents=True, exist_ok=True)

        self.best_score = None

        mlflow.start_run(run_name=f"run_{next_idx}")

    def end(self) -> None:
        """
        Финальное логирование параметров и метрик текущего трейла в MLflow.
        Завершение MLflow.

        :return:
        """
        self._check_flag_start()
        mlflow.end_run()
        self.flag_start = False

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
        self._check_flag_start()

        is_better = False
        if self.best_score is None:
            is_better = True
        elif self.direction == "maximize" and score > self.best_score:
            is_better = True
        elif self.direction == "minimize" and score < self.best_score:
            is_better = True

        if is_better:
            self.best_score = score
            filename = f"step_{step}_score={score:.5f}.png"
            filepath = self.current_example_dir / filename
            self._save_image(img, filepath)

        mlflow.log_params({k: str(v) for k, v in params.items()})
        mlflow.log_metric("score", score)
        mlflow.log_metric("step", step)

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

        # Нормализация к 0-255 и uint8, если нужно
        if img.max() <= 1.0:
            img = (img * 255).astype(np.uint8)
        elif img.dtype != np.uint8:
            img = img.astype(np.uint8)
        Image.fromarray(img).save(path)

    def _check_flag_start(self) -> None:
        if not self.flag_start:
            raise Exception("Logger not started. Call start() before.")
