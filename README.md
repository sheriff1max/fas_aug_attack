# Модуль для визуальных атак CV-моделей

## Проблематика

Модуль помогает найти такие условия съёмки изображений, при которых работа исследумой модели становится неусточивой, что влечёт за собой некорректные прогнозы.

С помощью инструментов модуля можно подобрать различные типы атак (добавление погодных условий, изменение цветов, создание размытия и множества других преобразований).

## Пример применения

Единственное, что нужно для запуска экспериментов - обернуть собственную модель в специальный класс, чтобы остальные интструменты модуля работали корректно. Например:

```python
from src.base import BaseModel
import torch
import torchvision


def get_transform(
    img_size: tuple[int] = (224, 224),
    normalize_mean: list[float] = [0.485, 0.456, 0.406],
    normalize_std: list[float] = [0.229, 0.224, 0.225],
):
    return torchvision.transforms.Compose(
        [
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=normalize_mean, std=normalize_std),
            torchvision.transforms.Resize(img_size),
        ]
    )


class Test_Model(BaseModel):
    def __init__(self, model: torch.nn.Module):
        super().__init__(model=model)

    def predict(self, img: Any) -> float:
        self.model.eval()
        with torch.no_grad():
            img = to_tensor(img).unsqueeze(0)
            img = img.to(device)
    
            val_results = self.model(img)
            val_output_list = val_results['similarity']

            # Вероятность класса, которую нужно "поломать".
            score = F.softmax(val_output_list, dim=-1).detach().cpu().numpy()[:, -1][0]
            return float(score)


to_tensor = get_transform()
model = Test_Model(model=net)
pred = model.predict(dataset[0]['img'])
# 0.02
```

Класс датасета можно либо реализоваться свой, либо можно взять из модуля (в таком случае нужно правильно расположить папки картинок).

> **Самое главное, чтобы ДАТАСЕТ возвращал словарь со значением 'img', которое хранит изображение в numpy формате!**

```python
from src.utils.dataset import AttackDataset

dataset = AttackDataset(
    path='/spoof-attack-liveness-face',
    exclude_folders=[],
)
print(f'Returns keys = {dataset[0].keys()}')
# Returns keys = dict_keys(['img', 'filename', 'path2file', 'is_real', 'type_attack'])
```

Далее нужно подобрать преобразования, на которых будет атакова модель для поиска её уязвимостей:

```python
list_type_transforms = [
    # Цветовые преобразования
    [
        transforms.BrightnessContrastTransform,
        transforms.HSVTransform,
        transforms.RGBShiftTransform,
        transforms.GammaTransform,
        transforms.SolarizeTransform,
        transforms.PosterizeTransform,
        transforms.EqualizeTransform,
        transforms.InvertTransform,
        transforms.ToGrayTransform,
        transforms.ChannelShuffleTransform,
        transforms.ToSepiaTransform,
        None,
    ],

    # Размытия
    [
        transforms.BlurTransform,
        transforms.GaussianBlurTransform,
        transforms.MedianBlurTransform,
        None,
    ],

    # Погодные условия
    [
        transforms.RainTransform,
        transforms.SnowTransform,
        None,
    ],

    # Изменение геометрии
    [
        transforms.PerspectiveTransform,
        transforms.ShiftScaleRotateTransform,
        None,
    ],

    # Сжатие изображения
    [
        transforms.CompressionTransform,
        transforms.DownscaleTransform,
        None,
    ],

    # Удаление областей
    [
        transforms.CoarseDropoutTransform,
        transforms.GridDropoutTransform,
        None,
    ],
]
```

Последним делом нужно настроить логгирование и Pipeline:

```python
from src.pipeline import PipelineAttackOptunaDataset
from src.utils.logging import LoggerOptuna

logger = LoggerOptuna(
    direction='maximize',
    description='Example run',
)

optuna_attack_pipeline_dataset = PipelineAttackOptunaDataset(
    model=model,
    list_type_transforms=list_type_transforms,
    logger=logger,
)

optuna_attack_pipeline_dataset.optimize(
    data=dataset,
    direction='maximize',
    n_trials=100,
    timeout=None,
    show_progress=True,
    catch=(ValueError, ),
)
```

## TODO:

- [x] Добавить Pipeline для всего датасета, а не одной img.
- [x] Добавить в класс Dataset загрузку конкретных доменов.
- Сделать BaseDataset. Сделать ResponseDataset
- Ограничить типы в BasePipelineAttackOptuna.optimize() на numpy | BaseDataset
- Добавить в OptunaPipelines метрику на данных без всяких преобразований, чтобы в будущем сравнить с оптимизациями.
- Обновить params importance
