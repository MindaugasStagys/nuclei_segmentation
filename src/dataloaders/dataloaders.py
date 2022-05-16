from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
from torchvision import transforms
from os.path import join
import numpy as np


from dataloaders.datasets import PanNukeDataset


class PanNukeDataModule(LightningDataModule):
    def __init__(self, data_dir: str, size: int, batch_size: int,
                 num_workers: int, train_fold: str, valid_fold: str,
                 test_fold: str):
        super().__init__()
        self.save_hyperparameters()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.size = size
        self.train_fold = train_fold
        self.valid_fold = valid_fold
        self.test_fold = test_fold
        self.test_df = self.ds_train = self.ds_valid = self.ds_test = None

    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            img_train = np.load(
                join(self.data_dir, 'images', f'{self.train_fold}.npy'),
                mmap_mode='r')
            img_valid = np.load(
                join(self.data_dir, 'images', f'{self.valid_fold}.npy'),
                mmap_mode='r')
            mask_train = np.load(
                join(self.data_dir, 'masks', f'{self.train_fold}.npy'),
                mmap_mode='r')
            mask_valid = np.load(
                join(self.data_dir, 'masks', f'{self.valid_fold}.npy'),
                mmap_mode='r')
            self.ds_train = PanNukeDataset(
                images=img_train,
                masks=mask_train,
                size=self.size,
                augment=True)
            self.ds_valid = PanNukeDataset(
                images=img_valid,
                masks=mask_valid,
                size=self.size,
                augment=False)

        if stage == 'test' or stage is None:
            img_test = np.load(
                join(self.data_dir, 'images', f'{self.test_fold}.npy'),
                mmap_mode='r')
            mask_test = np.load(
                join(self.data_dir, 'masks', f'{self.test_fold}.npy'),
                mmap_mode='r')
            self.ds_test = PanNukeDataset(
                images=img_test,
                masks=mask_test,
                size=self.size,
                augment=False)

    def train_dataloader(self):
        return DataLoader(self.ds_train,
                          batch_size=self.batch_size,
                          num_workers=self.num_workers,
                          shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.ds_valid,
                          batch_size=self.batch_size,
                          num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.ds_test,
                          batch_size=self.batch_size,
                          num_workers=self.num_workers)

