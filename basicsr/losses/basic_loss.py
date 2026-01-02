import torch
from torch import nn as nn
from torch.nn import functional as F

from basicsr.utils.registry import LOSS_REGISTRY
from .loss_util import weighted_loss

_reduction_modes = ['none', 'mean', 'sum']


@weighted_loss
def l1_loss(pred, target):
    return F.l1_loss(pred, target, reduction='none')


@weighted_loss
def mse_loss(pred, target):
    return F.mse_loss(pred, target, reduction='none')


@weighted_loss
def charbonnier_loss(pred, target, eps=1e-12):
    return torch.sqrt((pred - target)**2 + eps)


@LOSS_REGISTRY.register()
class L1Loss(nn.Module):
    """L1 (mean absolute error, MAE) loss.

    Args:
        loss_weight (float): Loss weight for L1 loss. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
    """

    def __init__(self, loss_weight=1.0, reduction='mean'):
        super(L1Loss, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. Supported ones are: {_reduction_modes}')

        self.loss_weight = loss_weight
        self.reduction = reduction

    def forward(self, pred, target, weight=None, **kwargs):
        """
        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, C, H, W). Ground truth tensor.
            weight (Tensor, optional): of shape (N, C, H, W). Element-wise weights. Default: None.
        """
        return self.loss_weight * l1_loss(pred, target, weight, reduction=self.reduction)


@LOSS_REGISTRY.register()
class MSELoss(nn.Module):
    """MSE (L2) loss.

    Args:
        loss_weight (float): Loss weight for MSE loss. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
    """

    def __init__(self, loss_weight=1.0, reduction='mean'):
        super(MSELoss, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. Supported ones are: {_reduction_modes}')

        self.loss_weight = loss_weight
        self.reduction = reduction

    def forward(self, pred, target, weight=None, **kwargs):
        """
        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, C, H, W). Ground truth tensor.
            weight (Tensor, optional): of shape (N, C, H, W). Element-wise weights. Default: None.
        """
        return self.loss_weight * mse_loss(pred, target, weight, reduction=self.reduction)


@LOSS_REGISTRY.register()
class CharbonnierLoss(nn.Module):
    """Charbonnier loss (one variant of Robust L1Loss, a differentiable
    variant of L1Loss).

    Described in "Deep Laplacian Pyramid Networks for Fast and Accurate
        Super-Resolution".

    Args:
        loss_weight (float): Loss weight for L1 loss. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
        eps (float): A value used to control the curvature near zero. Default: 1e-12.
    """

    def __init__(self, loss_weight=1.0, reduction='mean', eps=1e-12):
        super(CharbonnierLoss, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. Supported ones are: {_reduction_modes}')

        self.loss_weight = loss_weight
        self.reduction = reduction
        self.eps = eps

    def forward(self, pred, target, weight=None, **kwargs):
        """
        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, C, H, W). Ground truth tensor.
            weight (Tensor, optional): of shape (N, C, H, W). Element-wise weights. Default: None.
        """
        return self.loss_weight * charbonnier_loss(pred, target, weight, eps=self.eps, reduction=self.reduction)

    # def forward(self, pred, target, um, deg_app, weight=None, **kwargs):
    #     """
    #     Args:
    #         pred (Tensor): of shape (N, C, H, W). Predicted tensor.
    #         target (Tensor): of shape (N, C, H, W). Ground truth tensor.
    #         weight (Tensor, optional): of shape (N, C, H, W). Element-wise weights. Default: None.
    #         um: (2, 3*7, 256, 256)
    #         deg_app: # (2,12,7)
    #     """
    #     return self.loss_weight * charbonnier_loss(pred, target, um, deg_app, weight, eps=self.eps, reduction=self.reduction)


@LOSS_REGISTRY.register()
class WeightedTVLoss(L1Loss):
    """Weighted TV loss.

    Args:
        loss_weight (float): Loss weight. Default: 1.0.
    """

    def __init__(self, loss_weight=1.0, reduction='mean'):
        if reduction not in ['mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. Supported ones are: mean | sum')
        super(WeightedTVLoss, self).__init__(loss_weight=loss_weight, reduction=reduction)

    def forward(self, pred, weight=None):
        if weight is None:
            y_weight = None
            x_weight = None
        else:
            y_weight = weight[:, :, :-1, :]
            x_weight = weight[:, :, :, :-1]

        y_diff = super().forward(pred[:, :, :-1, :], pred[:, :, 1:, :], weight=y_weight)
        x_diff = super().forward(pred[:, :, :, :-1], pred[:, :, :, 1:], weight=x_weight)

        loss = x_diff + y_diff

        return loss
    

@LOSS_REGISTRY.register()
class BetaMSELoss(nn.Module):
    """MSE loss for beta vs deglist.

    Args:
        loss_weight (float): Loss weight for MSE loss. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Supported: 'none' | 'mean' | 'sum'. Default: 'mean'.
    """

    def __init__(self, loss_weight=1.0, reduction='mean'):
        super(BetaMSELoss, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}.')
        self.loss_weight = loss_weight
        self.reduction = reduction

    def forward(self, pred, target, weight=None, **kwargs):
        """
        pred: Tensor of shape (N, ...) — predicted beta
        target: Tensor of same shape — ground truth deglist
        weight: optional element-wise weight
        """
        loss = (pred - target) ** 2

        if weight is not None:
            loss = loss * weight

        if self.reduction == 'mean':
            loss = loss.mean()
        elif self.reduction == 'sum':
            loss = loss.sum()

        return self.loss_weight * loss
    
@LOSS_REGISTRY.register()
class PromptMetricLoss(nn.Module):
    r"""
    Label-aware metric loss for Static Prompt (SP) descriptors.

    L = sum_i [
        (1/|P(i)|) * sum_{j in P(i)} ||k_i_bar - k_j_bar||_2^2
      + (1/|N(i)|) * sum_{j in N(i)} [m - ||k_i_bar - k_j_bar||_2]_+^2
    ]

    Args:
        tau_pos (float): threshold tau_+ for positives (a_ij >= tau_pos).
        tau_neg (float): threshold tau_- for negatives (a_ij <= tau_neg).
        margin (float): hinge margin m for negatives.
        loss_weight (float): final multiplier.
        reduction (str): 'none' | 'mean' | 'sum'.
        eps (float): numerical stability.
    """
    def __init__(self,
                 tau_pos: float = 0.7,
                 tau_neg: float = 0.3,
                 margin: float = 0.5,
                 loss_weight: float = 1.0,
                 reduction: str = 'mean',
                 eps: float = 1e-8):
        super().__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}.')
        assert tau_pos > tau_neg, "Require tau_pos > tau_neg."
        self.tau_pos = float(tau_pos)
        self.tau_neg = float(tau_neg)
        self.margin = float(margin)
        self.loss_weight = float(loss_weight)
        self.reduction = reduction
        self.eps = float(eps)

    @staticmethod
    def _stack_descriptors(k):
        """
        Accept Tensor(S,d)/(S,*,d) or Dict[int, Tensor(...,d)] -> Tensor(S,d).
        Dict will be ordered by sorted keys.
        """
        if isinstance(k, dict):
            keys = sorted(k.keys())
            vs = []
            for kk in keys:
                v = k[kk]
                v = v.reshape(-1, v.shape[-1])
                v = v[0]  # take the first vector if more than one
                vs.append(v)
            k = torch.stack(vs, dim=0)
        elif isinstance(k, torch.Tensor):
            if k.ndim >= 3:
                k = k.reshape(k.shape[0], -1, k.shape[-1])[:, 0, :]
            elif k.ndim == 1:
                k = k.unsqueeze(0)
        else:
            raise TypeError("k must be Tensor or Dict[int, Tensor].")
        return k

    def forward(self, k, y, **kwargs):
        """
        Args:
            k: SP descriptors. Tensor (S,d) or Dict[int, Tensor(...,d)].
            y: multi-hot labels per *same* S items. Tensor (S,K).

        Returns:
            loss (scalar if reduction != 'none', else (S,))
        """
        k = self._stack_descriptors(k).float()            # (S,d)
        k_norm = torch.clamp(k.norm(p=2, dim=1, keepdim=True), min=self.eps)
        k_bar = k / k_norm                                 # (S,d)

        if not torch.is_tensor(y):
            raise TypeError("y must be a Tensor (S,K).")
        y = y.float()
        if y.shape[0] != k_bar.shape[0]:
            raise ValueError(f"y.shape[0] ({y.shape[0]}) != S ({k_bar.shape[0]}).")

        S = k_bar.shape[0]
        y_norm = torch.clamp(y.norm(p=2, dim=1, keepdim=True), min=self.eps)
        y_bar = y / y_norm
        affinity = y_bar @ y_bar.t()                       # (S,S)

        # masks
        eye = torch.eye(S, dtype=torch.bool, device=k.device)
        pos_mask = (affinity >= self.tau_pos) & (~eye)
        neg_mask = (affinity <= self.tau_neg) & (~eye)

        # pairwise distances on normalized vectors
        sim = k_bar @ k_bar.t()                            # (S,S)
        dist_sq = torch.clamp(2.0 - 2.0 * sim, min=0.0)
        dist = torch.sqrt(dist_sq + self.eps)

        # positives: mean squared distance
        pos_counts = pos_mask.sum(dim=1).clamp_min(1)
        pos_sum = (dist_sq * pos_mask.float()).sum(dim=1)
        pos_loss_i = torch.where(pos_mask.any(dim=1),
                                 pos_sum / pos_counts,
                                 torch.zeros_like(pos_sum))

        # negatives: mean squared hinge
        hinge = torch.clamp(self.margin - dist, min=0.0)
        neg_counts = neg_mask.sum(dim=1).clamp_min(1)
        neg_sum = ((hinge ** 2) * neg_mask.float()).sum(dim=1)
        neg_loss_i = torch.where(neg_mask.any(dim=1),
                                 neg_sum / neg_counts,
                                 torch.zeros_like(neg_sum))

        loss_i = pos_loss_i + neg_loss_i

        if self.reduction == 'mean':
            valid = (pos_mask.any(dim=1) | neg_mask.any(dim=1)).float()
            denom = valid.sum().clamp_min(1.0)
            loss = (loss_i * valid).sum() / denom
        elif self.reduction == 'sum':
            loss = loss_i.sum()
        else:
            loss = loss_i

        return self.loss_weight * loss
