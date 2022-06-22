import torch
import torch.nn.functional as F
from torch import nn


class ASPPConv(nn.Sequential):
    """ASPP convolution module."""
    def __init__(self, in_channels: int, out_channels: int, dilation: int):
        """
        Parameters
        ----------
        in_channels : int
            Number of input channels.
        out_channels : int
            Number of output channels.
        dilation : int
            Dilation rates for 2D convolution. Controls the spacing between the
            kernel points.
        """
        modules = [
            nn.Conv2d(in_channels, out_channels, 3, padding=dilation,
                      dilation=dilation, bias=False),
            nn.Mish(inplace=True),
            nn.BatchNorm2d(out_channels)
        ]
        super().__init__(*modules)


class ASPPPooling(nn.Sequential):
    """ASPP pooling module."""
    def __init__(self, in_channels: int, out_channels: int):
        """
        Parameters
        ----------
        in_channels : int
            Number of input channels.
        out_channels : int
            Number of output channels.
        """
        super().__init__(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.Mish(inplace=True),
            nn.BatchNorm2d(out_channels))

    def forward(self, x):
        """
        Parameters
        ----------
        x : torch.Tensor
            Input tensor.

        Returns
        -------
        torch.Tensor
            Output tensor after ASPP pooling operation.
        """
        size = x.shape[-2:]
        x = super().forward(x)
        return F.interpolate(x, size=size, mode='bilinear', align_corners=True)


class ASPP(nn.Module):
    """Atrous Spatial Pyramid Pooling."""
    def __init__(self, in_channels: int, out_channels: int, atrous_rates: list):
        """
        Parameters
        ----------
        in_channels : int
            Number of input channels.
        out_channels : int
            Number of output channels.
        atrous_rates : list
            A list of dilation rates for 2D convolution. Controls the spacing 
            between the kernel points.
        """
        super().__init__()
        modules = []
        modules.append(nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.Mish(inplace=True),
            nn.BatchNorm2d(out_channels)))
        for rate in atrous_rates:
            modules.append(ASPPConv(in_channels, out_channels, rate))
        modules.append(ASPPPooling(in_channels, out_channels))
        self.convs = nn.ModuleList(modules)
        self.project = nn.Sequential(
            nn.Conv2d(5 * out_channels, out_channels, 1, bias=False),
            nn.Mish(inplace=True),
            nn.BatchNorm2d(out_channels))

    def forward(self, x):
        """
        Parameters
        ----------
        x : torch.Tensor
            Input tensor.

        Returns
        -------
        torch.Tensor
            Output tensor after ASPP.
        """
        res = []
        for conv in self.convs:
            res.append(conv(x))
        res = torch.cat(res, dim=1)
        return self.project(res)


class AttentionBlock(nn.Module):
    """Attention block with learnable parameters"""
    def __init__(self, F_g: int, F_l: int, n_coefficients: int):
        """
        Parameters
        ----------
        F_g : int
            Number of feature maps (channels) in previous layer.
        F_l : int
            Number of feature maps in corresponding encoder layer transferred
            via skip connection.
        n_coefficients : int
            Number of learnable multi-dimensional attention coefficients.
        """
        super().__init__()
        self.W_gate = nn.Sequential(
            nn.Conv2d(F_g, n_coefficients, kernel_size=1, stride=1, 
                      padding=0, bias=True),
            nn.BatchNorm2d(n_coefficients)
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, n_coefficients, kernel_size=1, stride=1, 
                      padding=0, bias=True),
            nn.BatchNorm2d(n_coefficients)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(n_coefficients, 1, kernel_size=1, stride=1, 
                      padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, gate, skip_connection):
        """
        Parameters
        ----------
        gate : torch.Tensor
            Gating signal from previous layer.
        skip_connection : torch.Tensor
            Activation from corresponding encoder layer.

        Returns
        -------
        torch.Tensor
            Output activations.
        """
        g1 = self.W_gate(gate)
        x1 = self.W_x(skip_connection)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        out = skip_connection * psi
        return out

