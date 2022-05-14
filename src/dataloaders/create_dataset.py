from io import BytesIO
from urllib.request import urlopen
from zipfile import ZipFile
import shutil
import os


import requests


def download_data(url: str, save_path: str, chunk_size: int = 128):
    for i in range(1, 2):
        url_fold = f'{url}fold_{i}.zip'
        with urlopen(url_fold) as r:
            with ZipFile(BytesIO(r.read())) as z:
                z.extractall(save_path)


def move_files(data_dir: str):
    images_dir = os.path.join(data_dir, 'images')
    masks_dir = os.path.join(data_dir, 'masks')
    if not os.path.exists(images_dir):
        os.makedirs(images_dir)
    if not os.path.exists(masks_dir):
        os.makedirs(masks_dir)
    for fold in range(1, 2):
        img_file = os.path.join(
            data_dir, f'Fold {fold}', 'images', f'fold{fold}', 'images.npy')
        mask_file = os.path.join(
            data_dir, f'Fold {fold}', 'masks', f'fold{fold}', 'masks.npy')
        shutil.move(img_file, images_dir)
        shutil.move(mask_file, masks_dir)


def remove_directories(data_dir: str):
    for fold in range(1, 2):
        img_dir = os.path.join(data_dir, f'Fold {fold}')
        mask_dir = os.path.join(data_dir, f'Fold {fold}')
        shutil.rmtree(img_dir)
        shutil.rmtree(mask_dir)


if __name__ == '__main__':
    data_dir = os.path.abspath(os.path.join(__file__, '..', '..', '..', 'data'))
    images_dir = os.path.join(data_dir, 'images')
    masks_dir = os.path.join(data_dir, 'masks')
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    base_url = 'https://warwick.ac.uk/fac/cross_fac/tia/data/pannuke/'
    download_data(base_url, data_dir)
  #  move_files(data_dir)
  #  remove_directories(data_dir)

