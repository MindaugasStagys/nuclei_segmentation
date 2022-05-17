from torchtyping import TensorType, patch_typeguard
from typeguard import typechecked
from torch import nn, optim
import torch


patch_typeguard()


class Tversky(nn.Module):
    """Tversky loss."""
    @typechecked
    def __init__(self, beta: float = 0.3, eps: float = 1e-7):
        """
        Parameters
        ----------
        beta : float
            Weight of false negatives.
        eps: float
            A small constant added for numerical stability.
        """
        super().__init__()
        self.beta = beta
        self.eps = eps

    @typechecked
    def forward(self, 
                y_hat: TensorType['batch', 'N', 'size', 'size'], 
                y: TensorType['batch', 'N', 'size', 'size']):
        """
        Parameters
        ----------
        y_hat : torch.Tensor
            Tensor of predictions.
        y : torch.Tensor
            Tensor of ground truth.

        Returns
        -------
        torch.Tensor
            Tversky loss.
        """
        y_hat = y_hat.reshape(-1)
        y = y.reshape(-1)
        tp = (y_hat * y).sum()
        fp = ((1-y) * y_hat).sum()
        fn = (y * (1-y_hat)).sum()
        num = tp + self.eps
        denom = tp + (1-self.beta)*fp + self.beta*fn + self.eps
        return num / denom


class FocalTversky(nn.Module):
    """Focal Tversky loss."""
    @tpechecked
    def __init__(self, beta: float = 0.3, gamma: float = 4/3):
        """
        Parameters
        ----------
        beta : float
            Weight of false negatives.
        gamma: float
            Focusing parameter.
        """
        super().__init__()
        self.tversky = Tversky(beta=beta)
        self.gamma = gamma

    @typechecked
    def forward(self, 
                y_hat: TensorType['batch', 'N', 'size', 'size'], 
                y: TensorType['batch', 'N', 'size', 'size']):
        """
        Parameters
        ----------
        y_hat : torch.Tensor
            Tensor of predictions.
        y : torch.Tensor
            Tensor of ground truth.

        Returns
        -------
        torch.Tensor
            Focal Tversky loss.
        """
        tversky = self.tversky(y_hat, y)
        return (1 - tversky)**self.gamma

