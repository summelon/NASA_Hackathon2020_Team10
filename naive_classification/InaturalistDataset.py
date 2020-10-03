import torch

import os
import json
import random
import numpy as np
import pandas as pd
from PIL import Image


class InaturalistDataset(torch.utils.data.Dataset):
    def __init__(self, label_file, root_dir, transform, category_filter=''):
        with open(label_file, 'r') as json_file:
            self.label_anns = json.load(json_file)

        self.label_file_df = pd.merge(
            pd.DataFrame(self.label_anns['images'])[['id', 'file_name']].rename(columns={'id': 'image_id'}),
            pd.DataFrame(self.label_anns['annotations'])[['image_id', 'category_id']],
            on='image_id'
        )

        if isinstance(category_filter, list):
            self.label_file_df = self.label_file_df[self.label_file_df['category_id'].isin(category_filter)]
        elif isinstance(category_filter, str):
            self.label_file_df = self.label_file_df[self.label_file_df['file_name'].str.contains(category_filter)]

        self.label_file_df = self.label_file_df.reset_index().drop('index', axis=1)

        self.category_map = {}
        for i, category in enumerate(sorted(self.label_file_df['category_id'].unique())):
            self.category_map[category] = i

        self.invert_category_map = {value: key for key, value in self.category_map.items()}

        for key, value in self.category_map.items():
            self.label_file_df['category_id'] = self.label_file_df['category_id'].apply(lambda x: value if x == key else x)

        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.label_file_df)

    def __getitem__(self, idx):
        data = self.label_file_df.iloc[idx]

        image = Image.open(os.path.join(self.root_dir, data['file_name'])).convert('RGB')
        label = data['category_id']

        if self.transform:
            image = self.transform(image)

        return image, label

    def targets(self) -> int:
        return self.label_file_df['category_id'].nunique()

    def sample(self, weights) -> list:
        indices = [[], []]

        for key, value in weights.items():
            category = self.category_map[key]
            amount = sum(value)

            target = list(self.label_file_df[self.label_file_df['category_id'] == category].index)

            if amount > len(target):
                amount = len(target)

            temp = random.sample(list(self.label_file_df[self.label_file_df['category_id'] == category].index), k=amount)
            temp = np.random.permutation(temp)

            indices[0].extend(temp[: value[0]])
            indices[1].extend(temp[value[0]: ])

        return indices
    
    def labels(self) -> dict:
        return self.invert_category_map
