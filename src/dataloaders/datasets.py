from torch.utils.data import Dataset
from torchvision import transforms
import torch


from augmentations import normalize_stain_macenko


class PanNukeDataset(Dataset):
    def __init__(self, images, masks, size: int, crop: bool = False):
        self.images = images
        self.masks = masks
        if crop:
            self.transform_img = transforms.Compose([
                transforms.ToPILImage(),
                transforms.RandomCrop((size, size)),
                transforms.ToTensor()
            ])
        else:
            self.transform_img = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((size, size)),
                transforms.ToTensor()
            ])
        self.transform_mask = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((size, size)),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, ix: int):
        img = normalize_stain_macenko(self.images[ix])[0]
        img = self.transform_img(img)
        mask = torch.cat([
            self.transform_mask(self.masks[ix][:, :, x].astype(np.uint8))
            for x in range(0, self.masks[ix].shape[2])
        ])
        mask = mask.permute(2, 0, 1)
        return img, mask

