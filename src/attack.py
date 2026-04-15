import torch
from torchvision import transforms
from abc import ABC, abstractmethod


class BaseAttack(ABC):
    """Класс для создания атаки на модель с помощью
    различных преобразований входящих данных."""

    def __init__(
        self,
        model: torch.nn.Module
    ):
        self.model = model

    @abstractmethod
    def _predict(
        self,
        img: torch.tensor,
    ) -> float:
        """Предсказание вероятности положительного класса."""
        pass

    def predict(
        self,
        img: torch.tensor,
        transform: transforms.Compose,
    ) -> float:
        """Предсказание вероятности положительного класса."""
        img_new = transform(image=img)['image']
        return self._predict(img_new)
