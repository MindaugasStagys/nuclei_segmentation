from torchmetrics.functional import jaccard_index
from torch.utils.data import DataLoader
from typing_extensions import Literal
from typeguard import typechecked
from os.path import abspath, join
from typing import Optional
from sys import argv
import numpy as np
import torch
import yaml

from dataloaders.datasets import PredDataset
from dataloaders.dataloaders import PanNukeDataModule


@typechecked
def jaccard(pred_dl: DataLoader, target_dl: DataLoader, 
            n_classes: int, ignore_cls: int):
    list_inter = [[] for _ in range(n_classes)]
    list_union = [[] for _ in range(n_classes)]
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
def dice(pred_dl: DataLoader, target_dl: DataLoader, 
         n_classes: int, ignore_cls: int):
    list_inter = [[] for _ in range(n_classes)]
    list_denom = [[] for _ in range(n_classes)]
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


if __name__ == '__main__':
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
        masks = np.load(join(root, 'saved', 'preds', argv[1]))
        pred_dataloader = DataLoader(
            PredDataset(masks=masks),
            batch_size=config['data']['batch_size'],
            num_workers=config['data']['num_workers'])
        n_classes = config['data']['n_classes']

    test_dataloader.setup(stage='test')
    target_dataloader = test_dataloader.test_dataloader()

    iou = jaccard(
        pred_dl=pred_dataloader, 
        target_dl=target_dataloader,
        n_classes=n_classes,
        ignore_cls=5)
    print(f'Mean Jaccard score: {iou[1]}')
    print(f'Jaccard score per class: {list(iou[0])}')
    np.savetxt(join(root, 'saved', 'iou_per_class.out'), 
               iou[0], 
               delimiter=',')
    with open(join(root, 'saved', 'iou.out'), 'w') as f:
        f.write(str(iou[1].item()))

    dice = dice(
        pred_dl=pred_dataloader, 
        target_dl=target_dataloader,
        n_classes=n_classes,
        ignore_cls=5)
    print(f'Mean Dice score: {dice[1]}')
    print(f'Dice score per class: {list(dice[0])}')
    np.savetxt(join(root, 'saved', 'dice_per_class.out'), 
               dice[0], 
               delimiter=',')
    with open(join(root, 'saved', 'dice.out'), 'w') as f:
        f.write(str(dice[1].item()))


