from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
from torchvision import transforms
from os.path import join
import numpy as np


from datasets import PanNukeDataset


class PanNukeDataModule(LightningDataModule):
    def __init__(self, data_dir: str, batch_size: int, num_workers: int,
                 size: int, train_fold: str, valid_fold: str, test_fold: str):
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




import matplotlib.pyplot as plt

dataloader = PanNukeDataModule(
    data_dir='../../data',
    batch_size=16,
    num_workers=1,
    size=224, 
    train_fold='fold_1', 
    valid_fold='fold_2', 
    test_fold='fold_3')

dataloader.setup(stage='fit')
train_data = dataloader.train_dataloader()
ttt = next(iter(train_data))


mask = np.argmax(ttt[1][5], axis=0).numpy()
masked_data = np.ma.masked_where(mask == 5, mask)
plt.imshow(ttt[0][5].permute(1, 2, 0))
plt.imshow(masked_data, cmap='jet', alpha=0.5)
plt.show()

