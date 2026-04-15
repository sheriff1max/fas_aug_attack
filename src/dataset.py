from torch.utils.data import Dataset
from torchvision import transforms

import cv2

from pathlib import Path
import os


def get_transform(
    img_size: tuple[int] = (224, 224),
    normalize_mean: list[float] = [0.485, 0.456, 0.406],
    normalize_std: list[float] = [0.229, 0.224, 0.225],
):
    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=normalize_mean, std=normalize_std),
            transforms.Resize(img_size),
        ]
    )


class MyDataset(Dataset):
    """
    Dataset class for structure:

    - dataset_folder

        - domain_1
            - img_1.jpg
            - ...
            - img_N.jpg
        
        - ...
            - ...
        
        - domain_K
            - img_1.jpg
            - ...
            - img_M.jpg
    """
    def __init__(
        self,
        path: str,
        transform: transforms.Compose = None,
    ):
        self.path = Path(path)
        
        self.db = dict()
        i = 0
        for folder in os.listdir(path):
            if '.' not in folder:
                for filename in os.listdir(self.path / folder):
                    path2file = self.path / folder / filename
                    self.db[i] = path2file

                    i += 1

        self.transform = transform

    def __len__(self):
        return len(self.db)

    def __getitem__(self, idx):
        path2file = self.db[idx]
        # I have only negative classes in my dataset.
        is_real = 0
        # type_attack is name of domain folder.
        type_attack = str(self.db[idx]).split('/')[-2]
        
        img = cv2.imread(path2file)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        if self.transform is not None:
            img = self.transform(img)

        meta = {
            'path2file': str(path2file),
            'is_real': is_real,
            'type_attack': type_attack,
        }
        return img, meta
