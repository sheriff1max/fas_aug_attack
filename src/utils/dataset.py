from pathlib import Path
import os

import cv2
import numpy as np


class AttackDataset:
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

    :param path: путь до датасета с доменами атак
    :param exclude_folders: список названий доменов, которые исключить
    """
    def __init__(
        self,
        path: str,
        exclude_folders: list[str] = [],
    ):
        self.path = Path(path)
        
        self.db = dict()
        i = 0
        for folder in os.listdir(path):
            if '.' not in folder and folder not in exclude_folders:
                for filename in os.listdir(self.path / folder):
                    path2file = self.path / folder / filename
                    self.db[i] = path2file

                    i += 1

    def __len__(self):
        return len(self.db)

    def __getitem__(self, idx) -> dict:
        path2file = self.db[idx]
        # I have only negative classes in my dataset.
        is_real = 0
        # type_attack is name of domain folder.
        type_attack = path2file.parts[-2]
        filename = path2file.parts[-1]

        img: np.ndarray = cv2.imread(path2file)
        if img is None:
            raise ValueError(f'Path `{path2file}` is not exists.')
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        return {
            'img': img,
            'filename': filename,
            'path2file': str(path2file),
            'is_real': is_real,
            'type_attack': type_attack,
        }
