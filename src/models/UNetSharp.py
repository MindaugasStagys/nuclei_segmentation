from torchtyping import TensorType, patch_typeguard
from pytorch_lightning import LightningModule
from torch_optimizer import AdaBound
from typeguard import typechecked
from timm import create_model
from os.path import abspath, join
import torch.nn as nn
import torch
import yaml

from models.layers import ASPP, AttentionBlock
from models.losses import FocalTversky


patch_typeguard()

# Global parameters for typeguard
path = abspath(join(__file__, '..', '..', '..', 'configs', 'config.yaml'))
with open(path, 'r') as f:
    doc = yaml.safe_load(f)
    size = doc['data']['size']
    filters = doc['model']['filters']
    up_channels = filters[0] * 5
    n_classes = doc['data']['n_classes']


class UNetSharp(LightningModule):
    """UNetSharp model construction, training and inference methods."""
    def __init__(self, n_classes: int, filters: list, freeze_epochs: int,
                 optim_lr: float, optim_betas: tuple, optim_final_lr: float,
                 optim_gamma: float, optim_eps: float, optim_weight_decay: float,
                 optim_amsbound: bool, loss_beta: float, loss_gamma: float):
        """
        Parameters
        ----------
        n_classes : int
            Number of classes.
        filters : list
            List of depth values of each encoder block output.
        freeze_epochs : int
            For how many epochs should the backbone layers be freezed.
        optim_lr : float
            learning rate.
        optim_betas : tuple
            coefficients used for computing running averages of gradient and 
            its square.
        optim_final_lr : float
            Final (SGD) learning rate.
        optim_gamma : float
            Convergence speed of the bound functions.
        optim_eps : float
            Term added to the denominator to improve numerical stability.
        optim_weight_decay : float
            Weight decay (L2 penalty).
        optim_amsbound : bool
            Whether to use the AMSBound variant of this algorithm.
        loss_beta : float
            Weight of false negatives.
        loss_gamma: float
            Focusing parameter.
        """
        super().__init__()
        self.freeze_epochs = freeze_epochs

        # Encoder
        backbone_layers = self.get_backbone()
        self.en_block1 = nn.Sequential(*backbone_layers[:3])
        self.en_block2 = nn.Sequential(*backbone_layers[3:5])
        self.en_block3 = backbone_layers[5]
        self.en_block4 = backbone_layers[6]

        # Bridge
        self.br_block = backbone_layers[7]
        self.aspp = ASPP(filters[4], filters[0], [6, 12, 18])

        # Decoder block 4
        self.e1_d4_p = nn.MaxPool2d(8, 8, ceil_mode=True)
        self.e1_d4_conv = nn.Conv2d(filters[0], filters[0], 3, padding=1)
        self.e1_d4_act = nn.Mish(inplace=True)
        self.e1_d4_bn = nn.BatchNorm2d(filters[0])

        self.e2_d4_p = nn.MaxPool2d(4, 4, ceil_mode=True)
        self.e2_d4_conv = nn.Conv2d(filters[1], filters[0], 3, padding=1)
        self.e2_d4_act = nn.Mish(inplace=True)
        self.e2_d4_bn = nn.BatchNorm2d(filters[0])

        self.e3_d4_p = nn.MaxPool2d(2, 2, ceil_mode=True)
        self.e3_d4_conv = nn.Conv2d(filters[2], filters[0], 3, padding=1)
        self.e3_d4_act = nn.Mish(inplace=True)
        self.e3_d4_bn = nn.BatchNorm2d(filters[0])

        self.e4_d4_conv = nn.Conv2d(filters[3], filters[0], 3, padding=1)
        self.e4_d4_act = nn.Mish(inplace=True)
        self.e4_d4_bn = nn.BatchNorm2d(filters[0])

        self.br_d4_u = nn.Upsample(scale_factor=2, mode='bilinear')
        self.br_d4_conv = nn.Conv2d(filters[4], filters[0], 3, padding=1)
        self.br_d4_act = nn.Mish(inplace=True)
        self.br_d4_bn = nn.BatchNorm2d(filters[0])

        self.att_e4_br = AttentionBlock(filters[0], filters[0], filters[0])
        self.att_e3_br = AttentionBlock(filters[0], filters[0], filters[0])
        self.att_e2_br = AttentionBlock(filters[0], filters[0], filters[0])
        self.att_e1_br = AttentionBlock(filters[0], filters[0], filters[0])

        self.d4_conv = nn.Conv2d(up_channels, up_channels, 3, padding=1)
        self.d4_act = nn.Mish(inplace=True)
        self.d4_bn = nn.BatchNorm2d(up_channels)

        # Decoder block 3
        self.e1_d3_p = nn.MaxPool2d(4, 4, ceil_mode=True)
        self.e1_d3_conv = nn.Conv2d(filters[0], filters[0], 3, padding=1)
        self.e1_d3_act = nn.Mish(inplace=True)
        self.e1_d3_bn = nn.BatchNorm2d(filters[0])

        self.e2_d3_p = nn.MaxPool2d(2, 2, ceil_mode=True)
        self.e2_d3_conv = nn.Conv2d(filters[1], filters[0], 3, padding=1)
        self.e2_d3_act = nn.Mish(inplace=True)
        self.e2_d3_bn = nn.BatchNorm2d(filters[0])

        self.e3_d3_conv = nn.Conv2d(filters[2], filters[0], 3, padding=1)
        self.e3_d3_act = nn.Mish(inplace=True)
        self.e3_d3_bn = nn.BatchNorm2d(filters[0])

        self.d4_d3_u = nn.Upsample(scale_factor=2, mode='bilinear')
        self.d4_d3_conv = nn.Conv2d(up_channels, filters[0], 3, padding=1)
        self.d4_d3_act = nn.Mish(inplace=True)
        self.d4_d3_bn = nn.BatchNorm2d(filters[0])

        self.br_d3_u = nn.Upsample(scale_factor=4, mode='bilinear')
        self.br_d3_conv = nn.Conv2d(filters[4], filters[0], 3, padding=1)
        self.br_d3_act = nn.Mish(inplace=True)
        self.br_d3_bn = nn.BatchNorm2d(filters[0])

        self.att_e3_d4 = AttentionBlock(filters[0], filters[0], filters[0])
        self.att_e2_d4 = AttentionBlock(filters[0], filters[0], filters[0])
        self.att_e1_d4 = AttentionBlock(filters[0], filters[0], filters[0])

        self.d3_conv = nn.Conv2d(up_channels, up_channels, 3, padding=1)
        self.d3_act = nn.Mish(inplace=True)
        self.d3_bn = nn.BatchNorm2d(up_channels)

        # Decoder block 2
        self.e1_d2_p = nn.MaxPool2d(2, 2, ceil_mode=True)
        self.e1_d2_conv = nn.Conv2d(filters[0], filters[0], 3, padding=1)
        self.e1_d2_act = nn.Mish(inplace=True)
        self.e1_d2_bn = nn.BatchNorm2d(filters[0])

        self.e2_d2_conv = nn.Conv2d(filters[1], filters[0], 3, padding=1)
        self.e2_d2_act = nn.Mish(inplace=True)
        self.e2_d2_bn = nn.BatchNorm2d(filters[0])

        self.d3_d2_u = nn.Upsample(scale_factor=2, mode='bilinear')
        self.d3_d2_conv = nn.Conv2d(up_channels, filters[0], 3, padding=1)
        self.d3_d2_act = nn.Mish(inplace=True)
        self.d3_d2_bn = nn.BatchNorm2d(filters[0])

        self.d4_d2_u = nn.Upsample(scale_factor=4, mode='bilinear')
        self.d4_d2_conv = nn.Conv2d(up_channels, filters[0], 3, padding=1)
        self.d4_d2_act = nn.Mish(inplace=True)
        self.d4_d2_bn = nn.BatchNorm2d(filters[0])

        self.br_d2_u = nn.Upsample(scale_factor=8, mode='bilinear')
        self.br_d2_conv = nn.Conv2d(filters[4], filters[0], 3, padding=1)
        self.br_d2_act = nn.Mish(inplace=True)
        self.br_d2_bn = nn.BatchNorm2d(filters[0])

        self.att_e2_d3 = AttentionBlock(filters[0], filters[0], filters[0])
        self.att_e1_d3 = AttentionBlock(filters[0], filters[0], filters[0])

        self.d2_conv = nn.Conv2d(up_channels, up_channels, 3, padding=1)
        self.d2_act = nn.Mish(inplace=True)
        self.d2_bn = nn.BatchNorm2d(up_channels)

        # Decoder block 1
        self.e1_d1_conv = nn.Conv2d(filters[0], filters[0], 3, padding=1)
        self.e1_d1_act = nn.Mish(inplace=True)
        self.e1_d1_bn = nn.BatchNorm2d(filters[0])

        self.d2_d1_u = nn.Upsample(scale_factor=2, mode='bilinear')
        self.d2_d1_conv = nn.Conv2d(up_channels, filters[0], 3, padding=1)
        self.d2_d1_act = nn.Mish(inplace=True)
        self.d2_d1_bn = nn.BatchNorm2d(filters[0])

        self.d3_d1_u = nn.Upsample(scale_factor=4, mode='bilinear')
        self.d3_d1_conv = nn.Conv2d(up_channels, filters[0], 3, padding=1)
        self.d3_d1_act = nn.Mish(inplace=True)
        self.d3_d1_bn = nn.BatchNorm2d(filters[0])

        self.d4_d1_u = nn.Upsample(scale_factor=8, mode='bilinear')
        self.d4_d1_conv = nn.Conv2d(up_channels, filters[0], 3, padding=1)
        self.d4_d1_act = nn.Mish(inplace=True)
        self.d4_d1_bn = nn.BatchNorm2d(filters[0])

        self.br_d1_u = nn.Upsample(scale_factor=16, mode='bilinear')
        self.br_d1_conv = nn.Conv2d(filters[4], filters[0], 3, padding=1)
        self.br_d1_act = nn.Mish(inplace=True)
        self.br_d1_bn = nn.BatchNorm2d(filters[0])

        self.att_e1_d2 = AttentionBlock(filters[0], filters[0], filters[0])

        self.d1_conv = nn.Conv2d(up_channels, up_channels, 3, padding=1)
        self.d1_act = nn.Mish(inplace=True)
        self.d1_bn = nn.BatchNorm2d(up_channels)

        # output
        self.out_u = nn.Upsample(scale_factor=2, mode='bilinear')
        self.out_conv = nn.Conv2d(up_channels, n_classes, 3, padding=1)
        self.softmax = nn.Softmax(dim=1)

        # Optimizer
        self.optim_lr = optim_lr
        self.optim_betas = tuple(optim_betas)
        self.optim_final_lr = optim_final_lr
        self.optim_gamma = optim_gamma
        self.optim_eps = optim_eps
        self.optim_weight_decay = optim_weight_decay
        self.optim_amsbound = optim_amsbound
        
        # Loss
        self.loss = FocalTversky(beta=loss_beta, gamma=loss_gamma)

    @staticmethod
    def get_backbone():
        """Get pretrained encoder layers and freeze batch normalization"""
        backbone = create_model('swsl_resnext101_32x4d', pretrained=True)
        for m in backbone.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.weight.requires_grad_(False)
                m.bias.requires_grad_(False)
        return list(backbone.children())

    @typechecked
    def de_block_4(
        self, 
        e1: TensorType['batch', filters[0], int(size/2), int(size/2)],
        e2: TensorType['batch', filters[1], int(size/4), int(size/4)],
        e3: TensorType['batch', filters[2], int(size/8), int(size/8)],
        e4: TensorType['batch', filters[3], int(size/16), int(size/16)],
        br: TensorType['batch', filters[0], int(size/32), int(size/32)]
    ) -> TensorType['batch', up_channels, int(size/16), int(size/16)]:
        """4th decoder block

        Parameters
        ----------
        e1 : torch.Tensor
            Output of the 1st encoder block (skip connection).
        e2 : torch.Tensor
            Output of the 2nd encoder block (skip connection).
        e3 : torch.Tensor
            Output of the 3rd encoder block (skip connection).
        e4 : torch.Tensor
            Output of the 4th encoder block (skip connection).
        br : torch.Tensor
            Output of the bridge.

        Returns
        -------
        torch.Tensor
        """
        br_d4 = self.br_d4_bn(self.br_d4_act(self.br_d4_u(br)))
        e4_d4 = self.e4_d4_bn(self.e4_d4_act(self.e4_d4_conv(e4)))
        e4_d4 = self.att_e4_br(gate=br_d4, skip_connection=e4_d4)
        e3_d4 = self.e3_d4_bn(self.e3_d4_act(self.e3_d4_conv(self.e3_d4_p(e3))))
        e3_d4 = self.att_e3_br(gate=br_d4, skip_connection=e3_d4)
        e2_d4 = self.e2_d4_bn(self.e2_d4_act(self.e2_d4_conv(self.e2_d4_p(e2))))
        e2_d4 = self.att_e2_br(gate=br_d4, skip_connection=e2_d4)
        e1_d4 = self.e1_d4_bn(self.e1_d4_act(self.e1_d4_conv(self.e1_d4_p(e1))))
        e1_d4 = self.att_e1_br(gate=br_d4, skip_connection=e1_d4)
        d4 = self.d4_bn(self.d4_act(self.d4_conv(torch.cat(
            (e1_d4, e2_d4, e3_d4, e4_d4, br_d4), 1))))
        return d4
    
    @typechecked
    def de_block_3(
        self, 
        e1: TensorType['batch', filters[0], int(size/2), int(size/2)],
        e2: TensorType['batch', filters[1], int(size/4), int(size/4)],
        e3: TensorType['batch', filters[2], int(size/8), int(size/8)],
        d4: TensorType['batch', up_channels, int(size/16), int(size/16)], 
        br: TensorType['batch', filters[0], int(size/32), int(size/32)]
    ) -> TensorType['batch', up_channels, int(size/8), int(size/8)]:
        """3rd decoder block

        Parameters
        ----------
        e1 : torch.Tensor
            Output of the 1st encoder block (skip connection).
        e2 : torch.Tensor
            Output of the 2nd encoder block (skip connection).
        e3 : torch.Tensor
            Output of the 3rd encoder block (skip connection).
        d4 : torch.Tensor
            Output of the 4th decoder block.
        br : torch.Tensor
            Output of the bridge (skip connection).

        Returns
        -------
        torch.Tensor
        """
        d4_d3 = self.d4_d3_bn(self.d4_d3_act(self.d4_d3_conv(self.d4_d3_u(d4))))
        br_d3 = self.br_d3_bn(self.br_d3_act(self.br_d3_u(br)))
        e3_d3 = self.e3_d3_bn(self.e3_d3_act(self.e3_d3_conv(e3)))
        e3_d3 = self.att_e3_d4(gate=d4_d3, skip_connection=e3_d3)
        e2_d3 = self.e2_d3_bn(self.e2_d3_act(self.e2_d3_conv(self.e2_d3_p(e2))))
        e2_d3 = self.att_e2_d4(gate=d4_d3, skip_connection=e2_d3)
        e1_d3 = self.e1_d3_bn(self.e1_d3_act(self.e1_d3_conv(self.e1_d3_p(e1))))
        e1_d3 = self.att_e1_d4(gate=d4_d3, skip_connection=e1_d3)
        d3 = self.d3_bn(self.d3_act(self.d3_conv(torch.cat(
            (e1_d3, e2_d3, e3_d3, d4_d3, br_d3), 1))))
        return d3
    
    @typechecked
    def de_block_2(
        self, 
        e1: TensorType['batch', filters[0], int(size/2), int(size/2)],
        e2: TensorType['batch', filters[1], int(size/4), int(size/4)],
        d3: TensorType['batch', up_channels, int(size/8), int(size/8)], 
        d4: TensorType['batch', up_channels, int(size/16), int(size/16)], 
        br: TensorType['batch', filters[0], int(size/32), int(size/32)]
    ) -> TensorType['batch', up_channels, int(size/4), int(size/4)]:
        """2nd decoder block

        Parameters
        ----------
        e1 : torch.Tensor
            Output of the 1st encoder block (skip connection).
        e2 : torch.Tensor
            Output of the 2nd encoder block (skip connection).
        d3 : torch.Tensor
            Output of the 3rd decoder block.
        d4 : torch.Tensor
            Output of the 4th decoder block (skip connection).
        br : torch.Tensor
            Output of the bridge (skip connection).

        Returns
        -------
        torch.Tensor
        """
        d3_d2 = self.d3_d2_bn(self.d3_d2_act(self.d3_d2_conv(self.d3_d2_u(d3))))
        d4_d2 = self.d4_d2_bn(self.d4_d2_act(self.d4_d2_conv(self.d4_d2_u(d4))))
        br_d2 = self.br_d2_bn(self.br_d2_act(self.br_d2_u(br)))
        e2_d2 = self.e2_d2_bn(self.e2_d2_act(self.e2_d2_conv(e2)))
        e2_d2 = self.att_e2_d3(gate=d3_d2, skip_connection=e2_d2)
        e1_d2 = self.e1_d2_bn(self.e1_d2_act(self.e1_d2_conv(self.e1_d2_p(e1))))
        e1_d2 = self.att_e1_d3(gate=d3_d2, skip_connection=e1_d2)
        d2 = self.d2_bn(self.d2_act(self.d2_conv(torch.cat(
            (e1_d2, e2_d2, d3_d2, d4_d2, br_d2), 1))))
        return d2

    @typechecked
    def de_block_1(
        self, 
        e1: TensorType['batch', filters[0], int(size/2), int(size/2)],
        d2: TensorType['batch', up_channels, int(size/4), int(size/4)],
        d3: TensorType['batch', up_channels, int(size/8), int(size/8)], 
        d4: TensorType['batch', up_channels, int(size/16), int(size/16)], 
        br: TensorType['batch', filters[0], int(size/32), int(size/32)]
    ) -> TensorType['batch', up_channels, int(size/2), int(size/2)]:
        """3rd decoder block

        Parameters
        ----------
        e1 : torch.Tensor
            Output of the 1st encoder block (skip connection).
        d2 : torch.Tensor
            Output of the 2nd decoder block.
        d3 : torch.Tensor
            Output of the 3rd decoder block (skip connection).
        d4 : torch.Tensor
            Output of the 4th decoder block (skip connection).
        br : torch.Tensor
            Output of the bridge (skip connection).

        Returns
        -------
        torch.Tensor
        """
        d2_d1 = self.d2_d1_bn(self.d2_d1_act(self.d2_d1_conv(self.d2_d1_u(d2))))
        d3_d1 = self.d3_d1_bn(self.d3_d1_act(self.d3_d1_conv(self.d3_d1_u(d3))))
        d4_d1 = self.d4_d1_bn(self.d4_d1_act(self.d4_d1_conv(self.d4_d1_u(d4))))
        br_d1 = self.br_d1_bn(self.br_d1_act(self.br_d1_u(br)))
        e1_d1 = self.e1_d1_bn(self.e1_d1_act(self.e1_d1_conv(e1)))
        e1_d1 = self.att_e1_d2(gate=d2_d1, skip_connection=e1_d1)
        d1 = self.d1_bn(self.d1_act(self.d1_conv(torch.cat(
            (e1_d1, d2_d1, d3_d1, d4_d1, br_d1), 1))))
        return d1

    @typechecked
    def forward(self, x: TensorType['batch', 3, int(size), int(size)]) -> (
        TensorType['batch', n_classes], 
        TensorType['batch', n_classes, int(size), int(size)]
    ):
        """Forward method

        Parameters
        ----------
        x : torch.Tensor
            Batch of RGB images.

        Returns
        -------
        clss : torch.Tensor
            Output of classification branch.
        out : torch.Tensor
            Main output of semantic segmentation branch.
        """
        if self.trainer.current_epoch < self.freeze_epochs:
            with torch.no_grad():
                e1 = self.en_block1(x)
                e2 = self.en_block2(e1)
                e3 = self.en_block3(e2)
                e4 = self.en_block4(e3)
                br = self.br_block(e4)
        else:
            e1 = self.en_block1(x)
            e2 = self.en_block2(e1)
            e3 = self.en_block3(e2)
            e4 = self.en_block4(e3)
            br = self.br_block(e4)
        br = self.aspp(br)
        d4 = self.de_block_4(e1, e2, e3, e4, br)
        d3 = self.de_block_3(e1, e2, e3, d4, br)
        d2 = self.de_block_2(e1, e2, d3, d4, br)
        d1 = self.de_block_1(e1, d2, d3, d4, br)
        out = self.softmax(self.out_conv(self.out_u(d1)))
        return out

    def training_step(self, batch, batch_idx):
        img, mask = batch
        mask_pred = self(img)
        loss = self.loss(mask_pred, mask)
        self.log('train_loss', loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        img, mask = batch
        mask_pred = self(img)
        loss = self.loss(mask_pred, mask)
        self.log('val_loss', loss, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        img, mask = batch
        return self(img)

    def configure_optimizers(self):
        optimizer = AdaBound(
            [p for p in self.parameters() if p.requires_grad], 
            lr=self.optim_lr,
            betas=self.optim_betas,
            final_lr=self.optim_final_lr,
            gamma=self.optim_gamma,
            eps=self.optim_eps,
            weight_decay=self.optim_weight_decay,
            amsbound=self.optim_amsbound)
        return optimizer

