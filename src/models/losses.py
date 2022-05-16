from torch import nn, optim
from torchtyping import TensorType, patch_typeguard


patch_typeguard()


@typechecked
def FocalTverskyLoss(
    inputs: TensorType['batch', 'N', 'size', 'size'], 
    targets: TensorType['batch', 'N', 'size', 'size'], 
    alpha: float, beta: float = 0.3, gamma: float = 4/3, 
    smooth: float = 1e-6
):
    inputs = inputs.view(-1)
    targets = targets.view(-1)
    tp = (inputs * targets).sum()
    fp = ((1-targets) * inputs).sum()
    fn = (targets * (1-inputs)).sum()
    tversky = (tp + smooth) / (tp + alpha*fp + beta*fn + smooth)
    return (1 - tversky)**gamma

