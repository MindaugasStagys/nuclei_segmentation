from os.path import abspath, join
from sys import argv

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from torchmetrics.functional import jaccard_index
from typeguard import typechecked
from typing import Optional
from typing_extensions import Literal

from dataloaders.datasets import PredDataset
from dataloaders.dataloaders import PanNukeDataModule


@typechecked
def jaccard(pred_dl: DataLoader, target_dl: DataLoader,
            n_classes: int, ignore_cls: int):
    list_inter = list_union = [[] for _ in range(n_classes)]
    pred_iter = iter(pred_dl)
    for i, target in enumerate(target_dl):
        pred = next(pred_iter).argmax(1).view(-1).type(torch.uint8)
        target = target[1].argmax(1).view(-1).type(torch.uint8)
        for sem_class in range(n_classes):
            pred_ind = (pred == sem_class)
            target_ind = (target == sem_class)
            tmp_inter = pred_ind[target_ind].sum().item()
            tmp_union = pred_ind.sum().item() + \
                target_ind.sum().item() - \
                tmp_inter
            if tmp_union != 0:
                list_inter[sem_class].append(tmp_inter)
                list_union[sem_class].append(tmp_union)
    inter_cls_sum = [np.sum(inter) for inter in list_inter]
    union_cls_sum = [np.sum(union) for union in list_union]
    iou_per_class = np.divide(inter_cls_sum, union_cls_sum)
    inter_cls_sum = inter_cls_sum[:ignore_cls] + inter_cls_sum[(1+ignore_cls):]
    union_cls_sum = union_cls_sum[:ignore_cls] + union_cls_sum[(1+ignore_cls):]
    avg_iou = np.sum(inter_cls_sum) / np.sum(union_cls_sum)
    return iou_per_class, avg_iou


@typechecked
def jaccard_binary(pred_dl: DataLoader, target_dl: DataLoader, ignore_cls: int):
    inter_sum = union_sum = 0
    pred_iter = iter(pred_dl)
    for i, target in enumerate(target_dl):
        pred = next(pred_iter).argmax(1).view(-1).type(torch.uint8) + 1
        pred[pred == (ignore_cls+1)] = 0
        pred[pred > 0] = 1
        target = target[1].argmax(1).view(-1).type(torch.uint8) + 1
        target[target == (ignore_cls+1)] = 0
        target[target > 0] = 1
        pred_ind = (pred == 1)
        target_ind = (target == 1)
        tmp_inter = pred_ind[target_ind].sum().item()
        tmp_union = pred_ind.sum().item() + \
            target_ind.sum().item() - \
            tmp_inter
        inter_sum += tmp_inter
        union_sum += tmp_union
    return inter_sum / union_sum


@typechecked
def dice(pred_dl: DataLoader, target_dl: DataLoader,
         n_classes: int, ignore_cls: int):
    list_inter = list_denom = [[] for _ in range(n_classes)]
    pred_iter = iter(pred_dl)
    for i, target in enumerate(target_dl):
        pred = next(pred_iter).argmax(1).view(-1).type(torch.uint8)
        target = target[1].argmax(1).view(-1).type(torch.uint8)
        for sem_class in range(n_classes):
            pred_ind = (pred == sem_class)
            target_ind = (target == sem_class)
            tmp_inter = pred_ind[target_ind].sum().item()
            tmp_denom = pred_ind.sum().item() + target_ind.sum().item()
            if tmp_denom != 0:
                list_inter[sem_class].append(tmp_inter)
                list_denom[sem_class].append(tmp_denom)
    inter_cls_sum = [np.sum(inter) for inter in list_inter]
    denom_cls_sum = [np.sum(denom) for denom in list_denom]
    dice_per_class = np.divide(2.0*np.array(inter_cls_sum), denom_cls_sum)
    inter_cls_sum = inter_cls_sum[:ignore_cls] + inter_cls_sum[(1+ignore_cls):]
    denom_cls_sum = denom_cls_sum[:ignore_cls] + denom_cls_sum[(1+ignore_cls):]
    avg_dice = 2.0 * np.sum(inter_cls_sum) / np.sum(denom_cls_sum)
    return dice_per_class, avg_dice


@typechecked
def dice_binary(pred_dl: DataLoader, target_dl: DataLoader, ignore_cls: int):
    inter_sum = denom_sum = 0
    pred_iter = iter(pred_dl)
    for i, target in enumerate(target_dl):
        pred = next(pred_iter).argmax(1).view(-1).type(torch.uint8) + 1
        pred[pred == (ignore_cls+1)] = 0
        pred[pred > 0] = 1
        target = target[1].argmax(1).view(-1).type(torch.uint8) + 1
        target[target == (ignore_cls+1)] = 0
        target[target > 0] = 1
        pred_ind = (pred == 1)
        target_ind = (target == 1)
        inter_sum += pred_ind[target_ind].sum().item()
        denom_sum += pred_ind.sum().item() + target_ind.sum().item()
    return 2.0 * inter_sum / denom_sum


def main(pred_filename):
    root = abspath(join(__file__, '..', '..'))
    config_path = join(root, 'configs', 'config.yaml')
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
        test_dataloader = PanNukeDataModule(
            data_dir=join(root, 'data'),
            n_classes=config['data']['n_classes'],
            size=config['data']['size'],
            train_fold=config['data']['train_fold'],
            valid_fold=config['data']['valid_fold'],
            test_fold=config['data']['test_fold'],
            batch_size=config['data']['batch_size'],
            num_workers=config['data']['num_workers'])
        masks = np.load(join(root, 'saved', 'preds', pred_filename))
        pred_dataloader = DataLoader(
            PredDataset(masks=masks),
            batch_size=config['data']['batch_size'],
            num_workers=config['data']['num_workers'])
        n_classes = config['data']['n_classes']
        ignore_cls = config['data']['background_cls']

    test_dataloader.setup(stage='test')
    target_dataloader = test_dataloader.test_dataloader()

    iou = jaccard(
        pred_dl=pred_dataloader,
        target_dl=target_dataloader,
        n_classes=n_classes,
        ignore_cls=ignore_cls)
    iou_binary = jaccard_binary(
        pred_dl=pred_dataloader,
        target_dl=target_dataloader,
        ignore_cls=ignore_cls)
    print(f'Mean Jaccard score: {iou[1]}')
    print(f'Jaccard score per class: {list(iou[0])}')
    print(f'Binary Jaccard score: {iou_binary}')
    np.savetxt(join(root, 'saved', 'iou_per_class.out'),
               iou[0],
               delimiter=',')
    with open(join(root, 'saved', 'iou_average.out'), 'w') as f:
        f.write(str(iou[1].item()))
    with open(join(root, 'saved', 'iou_binary.out'), 'w') as f:
        f.write(str(iou_binary))

    dice = dice(
        pred_dl=pred_dataloader,
        target_dl=target_dataloader,
        n_classes=n_classes,
        ignore_cls=ignore_cls)
    dice_binary = dice_binary(
        pred_dl=pred_dataloader,
        target_dl=target_dataloader,
        ignore_cls=ignore_cls)
    print(f'Mean Dice score: {dice[1]}')
    print(f'Dice score per class: {list(dice[0])}')
    print(f'Binary Dice score: {dice_binary}')
    np.savetxt(join(root, 'saved', 'dice_per_class.out'),
               dice[0],
               delimiter=',')
    with open(join(root, 'saved', 'dice_average.out'), 'w') as f:
        f.write(str(dice[1].item()))
    with open(join(root, 'saved', 'dice_binary.out'), 'w') as f:
        f.write(str(dice_binary))


if __name__ == '__main__':
    main(argv[1])

