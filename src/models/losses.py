from torchtyping import TensorType, patch_typeguard
from typeguard import typechecked
from torch import nn, optim


patch_typeguard()


@typechecked
class TverskyLoss:
    """Tversky and Focal Tversky loss functions."""
    def __init__(self, alpha: float, beta: float = 0.3, 
                 gamma: float = 4/3, smooth: float = 1e-6):
        """
        Parameters
        ----------
        alpha : float
            Weight of false positives.
        beta : float
            Weight of false negatives.
        gamma: float
            Focusing parameter.
        smooth: float
            A small constant added to avoid zero and nan.
        """
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.smooth = smooth

    @typechecked
    def tversky(self, 
                inputs: TensorType['batch', 'N', 'size', 'size'], 
                targets: TensorType['batch', 'N', 'size', 'size']):
        """Calculate Tversky loss function.

        Parameters
        ----------
        inputs : torch.Tensor
            Tensor of predictions.
        targets : torch.Tensor
            Tensor of ground truth.

        Returns
        -------
        torch.Tensor
            Tversky loss.
        """
        inputs = inputs.reshape(-1)
        targets = targets.reshape(-1)
        tp = (inputs * targets).sum()
        fp = ((1-targets) * inputs).sum()
        fn = (targets * (1-inputs)).sum()
        tversky = (tp + self.smooth) / \
            (tp + self.alpha*fp + self.beta*fn + self.smooth)
        return tversky

    @typechecked
    def focal_tversky(self, 
                      inputs: TensorType['batch', 'N', 'size', 'size'], 
                      targets: TensorType['batch', 'N', 'size', 'size']):
        """Calculate Focal Tversky loss function.

        Parameters
        ----------
        inputs : torch.Tensor
            Tensor of predictions.
        targets : torch.Tensor
            Tensor of ground truth.

        Returns
        -------
        torch.Tensor
            Focal Tversky loss.
        """
        tversky = self.tversky(inputs, targets)
        return (1 - tversky)**self.gamma

