from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np
import torch


from dataloaders.augmentations.stain_normalization import normalize_stain_macenko
from dataloaders.augmentations.transforms import RandomCrop, aug_random


class PanNukeDataset(Dataset):
    def __init__(self, images, masks, size: int, augment: bool = False):
        self.images = images
        self.masks = masks
        self.augment = augment
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((size, size)),
            transforms.ToTensor()
        ])
        self.random_crop = RandomCrop(size, size)
        self.aug_random = aug_random()

    def __len__(self):
        return len(self.images)

    def __getitem__(self, ix: int):
        img = self.images[ix]
        mask = self.masks[ix]
        img = normalize_stain_macenko(img)[0]
        if self.augment:
            data = {'image': img, 'mask': mask}
            data['image'], data['mask'] = self.random_crop.apply(
                img=data['image'],
                mask=data['mask'])
            data = self.aug_random(**data)
            img = torch.from_numpy(data['image']).permute(2, 0, 1)
            mask = torch.from_numpy(data['mask']).permute(2, 0, 1)
        else:
            img = self.transform(img)
            mask = torch.cat([
                self.transform(self.masks[ix][:, :, x].astype(np.uint8))
                for x in range(0, self.masks[ix].shape[2])
            ])
        return img, mask

