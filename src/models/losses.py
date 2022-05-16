from torchtyping import TensorType, patch_typeguard
from typeguard import typechecked
from torch import nn, optim


patch_typeguard()


@typechecked
def FocalTverskyLoss(
    inputs: TensorType['batch', 'N', 'size', 'size'], 
    targets: TensorType['batch', 'N', 'size', 'size'], 
    alpha: float, beta: float = 0.3, gamma: float = 4/3, 
    smooth: float = 1e-6):
    """Focal Tversky loss function.

    Parameters
    ----------
    inputs : torch.Tensor
        Tensor of predictions.
    targets : torch.Tensor
        Tensor of ground truth.
    loss_alpha : float
        Weight of false positives.
    loss_beta : float
        Weight of false negatives.
    loss_gamma: float
        Focusing parameter.
    loss_smooth: float
        A small constant added to avoid zero and nan.
    """
    inputs = inputs.view(-1)
    targets = targets.view(-1)
    tp = (inputs * targets).sum()
    fp = ((1-targets) * inputs).sum()
    fn = (targets * (1-inputs)).sum()
    tversky = (tp + smooth) / (tp + alpha*fp + beta*fn + smooth)
    return (1 - tversky)**gamma

