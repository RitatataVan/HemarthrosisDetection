import pytorch_lightning as pl
from torchvision import transforms
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
import torch

import pandas as pd
import numpy as np
from PIL import Image

from hyperparameters import parameters as params

class NovoDataset(Dataset):
    def __init__(self, ids, labels, transf):
        super().__init__()

        # Transforms
        self.transforms = transf

        # Images IDS amd Labels
        self.ids = ids
        self.labels = labels

    def __getitem__(self, index):
        # Get an ID of a specific image
        id_img = self.ids[index]

        # Open Image
        img = Image.open(id_img).convert('RGB')
        img = self.transforms(img)
        img = img.type(torch.float32)

        # Get Label
        label = torch.as_tensor(self.labels[index], dtype=torch.float16)

        return img, label

    def __len__(self):
        return len(self.ids)

class NovoDataModule(pl.LightningDataModule):

    def __init__(self):
        super().__init__()
        self.batch_size = params['batch_size']
        self.num_workers = params['num_workers']
        self.transform = transforms.Compose([
            transforms.Resize((params['img_size'], params['img_size'])),
            transforms.ToTensor(),
            transforms.Normalize(params['data_mean'], params['data_std'])])

    def setup(self, stage=None):
        # we set up only relevant datasets when stage is specified
        if stage == 'fit' or stage is None:
            images = pd.read_csv(params['train_data'])
            self.novo_train = NovoDataset(ids=images['ID_IMG'].tolist(), 
                                        labels=images['BLOOD'].tolist(), transf=self.transform)
        if stage == 'test' or stage is None:
            images = pd.read_csv(params['test_data'])
            self.novo_test = NovoDataset(ids=images['ID_IMG'].tolist(),
                                        labels=images['BLOOD'].tolist(), transf=self.transform)

    def train_dataloader(self):
        novo_train = DataLoader(self.novo_train, batch_size=self.batch_size,
                                num_workers=self.num_workers, pin_memory=True)
        return novo_train

    def test_dataloader(self):
        novo_test = DataLoader(self.novo_test, batch_size=self.batch_size,
                                num_workers=self.num_workers, pin_memory=True)
        return novo_test