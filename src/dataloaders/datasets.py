import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms

from dataloaders.augmentations.stain_normalization import normalize_stain_macenko
from dataloaders.augmentations.transforms import RandomCrop, aug_random


class PanNukeDataset(Dataset):
    def __init__(self, images, masks, n_classes: int, 
                 size: int, augment: bool = False):
        self.images = images
        self.masks = masks
        self.n_classes = n_classes
        self.augment = augment
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((size, size)),
            transforms.ToTensor()
        ])
        self.random_crop = RandomCrop(size, size)
        self.aug_random = aug_random()
        self.to_tensor = transforms.ToTensor()

    def __len__(self):
        return len(self.images)

    def __getitem__(self, ix: int):
        img = np.array(self.images[ix])
        mask = np.array(self.masks[ix])
        img = normalize_stain_macenko(img)[0]
        if self.augment:
            data = {'image': img, 'mask': mask}
            data['image'], data['mask'] = self.random_crop.apply(
                img=data['image'],
                mask=data['mask'])
            data = self.aug_random(**data)
            img = self.to_tensor(data['image'])
            mask = self.to_tensor(data['mask'])
        else:
            img = self.transform(img)
            mask = torch.cat([
                self.transform(self.masks[ix][:, :, i].astype(np.uint8))
                for i in range(0, self.masks[ix].shape[2])
            ])
        one_hot = torch.zeros(self.n_classes, mask.shape[1], mask.shape[2])
        mask = np.argmax(mask, axis=0)
        for i in range(0, self.n_classes):
            one_hot[i, :, :][mask == i] = 1
        return img, one_hot


class PanNukeDatasetMasksOnly(Dataset):
    def __init__(self, masks, n_classes: int, size: int):
        self.masks = masks
        self.n_classes = n_classes
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((size, size)),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.masks)

    def __getitem__(self, ix: int):
        mask = np.array(self.masks[ix])
        mask = torch.cat([
            self.transform(self.masks[ix][:, :, i].astype(np.uint8))
            for i in range(0, self.masks[ix].shape[2])
        ])
        one_hot = torch.zeros(self.n_classes, mask.shape[1], mask.shape[2])
        mask = np.argmax(mask, axis=0)
        for i in range(0, self.n_classes):
            one_hot[i, :, :][mask == i] = 1
        return one_hot


class PredDataset(Dataset):
    def __init__(self, masks):
        self.masks = masks

    def __len__(self):
        return len(self.masks)

    def __getitem__(self, ix: int):
        mask = torch.tensor(self.masks[ix])
        return mask

