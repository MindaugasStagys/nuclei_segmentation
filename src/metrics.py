from torchmetrics.functional import jaccard_index
from torch.utils.data import DataLoader
from typing_extensions import Literal
from typeguard import typechecked
from os.path import abspath, join
from typing import Optional
from sys import argv
import numpy as np
import pandas as pd
import torch
import yaml

from dataloaders.datasets import PanNukeDatasetMasksOnly, PredDataset
from dataloaders.dataloaders import PanNukeDataModule


@typechecked
def jaccard(pred_dl: DataLoader, target_dl: DataLoader,
            n_classes: int, ignore_cls: int):
    list_inter = [[] for _ in range(n_classes)]
    list_union = [[] for _ in range(n_classes)]
    pred_iter = iter(pred_dl)
    for i, target in enumerate(target_dl):
        pred = next(pred_iter).argmax(1).view(-1).type(torch.uint8)
        target = target.argmax(1).view(-1).type(torch.uint8)
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
    inter_sum = 0
    union_sum = 0
    pred_iter = iter(pred_dl)
    for i, target in enumerate(target_dl):
        pred = next(pred_iter).argmax(1).view(-1).type(torch.uint8) + 1
        pred[pred == (ignore_cls+1)] = 0
        pred[pred > 0] = 1
        target = target.argmax(1).view(-1).type(torch.uint8) + 1
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
    list_inter = [[] for _ in range(n_classes)]
    list_denom = [[] for _ in range(n_classes)]
    pred_iter = iter(pred_dl)
    for i, target in enumerate(target_dl):
        pred = next(pred_iter).argmax(1).view(-1).type(torch.uint8)
        target = target.argmax(1).view(-1).type(torch.uint8)
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
    inter_sum = 0
    denom_sum = 0
    pred_iter = iter(pred_dl)
    for i, target in enumerate(target_dl):
        pred = next(pred_iter).argmax(1).view(-1).type(torch.uint8) + 1
        pred[pred == (ignore_cls+1)] = 0
        pred[pred > 0] = 1
        target = target.argmax(1).view(-1).type(torch.uint8) + 1
        target[target == (ignore_cls+1)] = 0
        target[target > 0] = 1
        pred_ind = (pred == 1)
        target_ind = (target == 1)
        inter_sum += pred_ind[target_ind].sum().item()
        denom_sum += pred_ind.sum().item() + target_ind.sum().item()
    return 2.0 * inter_sum / denom_sum


def main():
    root = abspath(join(__file__, '..', '..'))
    config_path = join(root, 'configs', 'config.yaml')
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
        n_classes = config['data']['n_classes']
        size = config['data']['size']
        test_fold = config['data']['test_fold']
        batch_size = config['data']['batch_size']
        num_workers = config['data']['num_workers']
    ignore_cls = 5
    cols = ['Neoplastic', 'Inflammatory', 'Connective', 
            'Dead', 'Epithelial', 'Average', 'Binary']

    test_masks = np.load(join(root, 'data', 'masks', f'{test_fold}.npy'), 
                         mmap_mode='r')
    pred_masks = np.load(join(root, 'saved', 'preds', 'preds_33.npy'),
                         mmap_mode='r')
    types = np.load(join(root, 'data', 'types', f'{test_fold}.npy'),
                    mmap_mode='r')

    ds_test = PanNukeDatasetMasksOnly(
        masks=test_masks,
        n_classes=n_classes,
        size=size)
    target_dataloader = DataLoader(
        ds_test,
        batch_size=batch_size,
        num_workers=num_workers)
    pred_dataloader = DataLoader(
        PredDataset(masks=pred_masks),
        batch_size=batch_size,
        num_workers=num_workers)

    iou = jaccard(
        pred_dl=pred_dataloader,
        target_dl=target_dataloader,
        n_classes=n_classes,
        ignore_cls=ignore_cls)
    iou_binary = jaccard_binary(
        pred_dl=pred_dataloader,
        target_dl=target_dataloader,
        ignore_cls=ignore_cls)
    list_iou_all = list(iou[0][:ignore_cls]) + \
        list(iou[0][(1 + ignore_cls):]) + [iou[1]] + [iou_binary]

    coef_dice = dice(
        pred_dl=pred_dataloader,
        target_dl=target_dataloader,
        n_classes=n_classes,
        ignore_cls=ignore_cls)
    coef_dice_binary = dice_binary(
        pred_dl=pred_dataloader,
        target_dl=target_dataloader,
        ignore_cls=ignore_cls)
    list_dice_all = list(coef_dice[0][:ignore_cls]) + \
        list(coef_dice[0][(1 + ignore_cls):]) + \
        [coef_dice[1]] + [coef_dice_binary]

    # CALCULATE METRICS BY TISSUE ---------------------------------------------

    unique_types = np.unique(types)
    test_masks_by_group = [test_masks[types == i] for i in unique_types]
    preds_by_group = [pred_masks[types == i] for i in unique_types]
    
    iou_by_group = []
    dice_by_group = []

    for i, v in enumerate(unique_types):
        ds_test = PanNukeDatasetMasksOnly(
            masks=test_masks_by_group[i],
            n_classes=n_classes,
            size=size)
        target_dataloader = DataLoader(
            ds_test,
            batch_size=batch_size,
            num_workers=num_workers)
        pred_dataloader = DataLoader(
            PredDataset(masks=preds_by_group[i]),
            batch_size=batch_size,
            num_workers=num_workers)
        
        iou = jaccard(
            pred_dl=pred_dataloader,
            target_dl=target_dataloader,
            n_classes=n_classes,
            ignore_cls=ignore_cls)
        iou_binary = jaccard_binary(
            pred_dl=pred_dataloader,
            target_dl=target_dataloader,
            ignore_cls=ignore_cls)
        list_iou = list(iou[0][:ignore_cls]) + \
            list(iou[0][(1 + ignore_cls):]) + [iou[1]] + [iou_binary]
        iou_by_group.append(list_iou)

        coef_dice = dice(
            pred_dl=pred_dataloader,
            target_dl=target_dataloader,
            n_classes=n_classes,
            ignore_cls=ignore_cls)        
        coef_dice_binary = dice_binary(
            pred_dl=pred_dataloader,
            target_dl=target_dataloader,
            ignore_cls=ignore_cls)
        list_dice = list(coef_dice[0][:ignore_cls]) + \
            list(coef_dice[0][(1 + ignore_cls):]) + \
            [coef_dice[1]] + [coef_dice_binary]
        dice_by_group.append(list_dice)

    iou_by_group.append(list_iou_all)
    dice_by_group.append(list_dice_all)
    index = list(unique_types)
    index.extend(['Average'])
    df_iou = pd.DataFrame(iou_by_group, columns=cols, index=index)
    df_dice = pd.DataFrame(dice_by_group, columns=cols, index=index)
    df_iou.to_csv(join(root, 'saved', 'iou.csv'))
    df_dice.to_csv(join(root, 'saved', 'dice.csv'))


if __name__ == '__main__':
    main()

