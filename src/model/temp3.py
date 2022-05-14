import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
import torch

from preprocessing import normalize_stain_macenko

size = 224
filters = [64, 256, 512, 1024, 2048]
up_channels = filters[0] * 5
n_classes = 2


transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225])
])

transform2 = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

model = UNetSharp()
model.eval()
dummy_input = torch.ones((4, 3, 224, 224))
dummy_output = model(dummy_input)

path = "../../data/Fold 1/images/fold1/images.npy"
temp = np.load(path, mmap_mode='r')

img = torch.tensor(temp[0]).permute(2,0,1)



path2 = "../../data/Fold 1/images/fold1/types.npy"
temp2 = np.load(path2, mmap_mode='r')

path3 = "../../data/Fold 1/masks/fold1/masks.npy"
temp3 = np.load(path3, mmap_mode='r')

ttt = normalize_stain_macenko(temp[500])[0]
img2 = transform(temp3[200][:,:,5].astype(np.uint8))
img2 = topil(img2)
img2.show()

img = normalize_stain_macenko(temp[200])[0]
img = transform2(img).permute(1, 2, 0).numpy()
mask = torch.cat(
    [transform2(temp3[200][:, :, x].astype(np.uint8)) 
     for x in range(0, temp3[200].shape[2])]
)
mask = np.argmax(mask, axis=0)
masked_data = np.ma.masked_where(mask == 5, mask)
plt.imshow((img * 255).astype(np.uint8))
plt.imshow(masked_data, cmap='jet', alpha=0.5)
plt.show()

plt.imshow(mask, cmap='nipy_spectral')


x = transform(x.astype(np.uint8))
y = torch.cat([x, x])

ttt = temp3[200][:, :, 2].tolist()
ttt = [t for sublist in ttt for t in sublist]
np.unique(ttt)
