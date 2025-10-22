"""
Helpers for various likelihood-based losses. These are ported from the original
Ho et al. diffusion models codebase:
https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/utils.py
"""

import numpy as np

import torch as th


def normal_kl(mean1, logvar1, mean2, logvar2):
    """
    Compute the KL divergence between two gaussians.

    Shapes are automatically broadcasted, so batches can be compared to
    scalars, among other use cases.
    """
    tensor = None
    for obj in (mean1, logvar1, mean2, logvar2):
        if isinstance(obj, th.Tensor):
            tensor = obj
            break
    assert tensor is not None, "at least one argument must be a Tensor"

    # Force variances to be Tensors. Broadcasting helps convert scalars to
    # Tensors, but it does not work for th.exp().
    logvar1, logvar2 = [
        x if isinstance(x, th.Tensor) else th.tensor(x).to(tensor)
        for x in (logvar1, logvar2)
    ]

    return 0.5 * (
        -1.0
        + logvar2
        - logvar1
        + th.exp(logvar1 - logvar2)
        + ((mean1 - mean2) ** 2) * th.exp(-logvar2)
    )


def approx_standard_normal_cdf(x):
    """
    A fast approximation of the cumulative distribution function of the
    standard normal.
    """
    return 0.5 * (1.0 + th.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * th.pow(x, 3))))


def discretized_gaussian_log_likelihood(x, *, means, log_scales):
    """
    Compute the log-likelihood of a Gaussian distribution discretizing to a
    given image.

    :param x: the target images. It is assumed that this was uint8 values,
              rescaled to the range [-1, 1].
    :param means: the Gaussian mean Tensor.
    :param log_scales: the Gaussian log stddev Tensor.
    :return: a tensor like x of log probabilities (in nats).
    """
    assert x.shape == means.shape == log_scales.shape
    centered_x = x - means
    inv_stdv = th.exp(-log_scales)
    plus_in = inv_stdv * (centered_x + 1.0 / 255.0)
    cdf_plus = approx_standard_normal_cdf(plus_in)
    min_in = inv_stdv * (centered_x - 1.0 / 255.0)
    cdf_min = approx_standard_normal_cdf(min_in)
    log_cdf_plus = th.log(cdf_plus.clamp(min=1e-12))
    log_one_minus_cdf_min = th.log((1.0 - cdf_min).clamp(min=1e-12))
    cdf_delta = cdf_plus - cdf_min
    log_probs = th.where(
        x < -0.999,
        log_cdf_plus,
        th.where(x > 0.999, log_one_minus_cdf_min, th.log(cdf_delta.clamp(min=1e-12))),
    )
    assert log_probs.shape == x.shape
    return log_probs


# ==================== MedSegDiff-V2 with Classification ====================
# Enhanced loss functions for joint segmentation and classification

def dice_loss(pred, target, smooth=1e-5):
    """
    Dice Loss for segmentation
    
    Args:
        pred: predicted segmentation logits/probabilities [B, C, H, W]
        target: ground truth segmentation masks [B, C, H, W]
        smooth: smoothing factor to avoid division by zero
    
    Returns:
        dice_loss: 1 - Dice coefficient
    """
    # Apply sigmoid if pred is logits
    if pred.min() < 0 or pred.max() > 1:
        pred = th.sigmoid(pred)
    
    # Flatten spatial dimensions
    pred_flat = pred.view(pred.size(0), pred.size(1), -1)  # [B, C, H*W]
    target_flat = target.view(target.size(0), target.size(1), -1)  # [B, C, H*W]
    
    # Compute intersection and union
    intersection = (pred_flat * target_flat).sum(dim=2)  # [B, C]
    pred_sum = pred_flat.sum(dim=2)  # [B, C]
    target_sum = target_flat.sum(dim=2)  # [B, C]
    
    # Dice coefficient
    dice = (2.0 * intersection + smooth) / (pred_sum + target_sum + smooth)
    
    # Return mean Dice loss across batch and classes
    return 1.0 - dice.mean()


def focal_loss(pred, target, alpha=0.25, gamma=2.0, reduction='mean'):
    """
    Focal Loss for handling class imbalance
    
    Args:
        pred: predicted logits [B, C, H, W] or [B, num_classes]
        target: ground truth labels [B, C, H, W] or [B]
        alpha: weighting factor for balanced loss
        gamma: focusing parameter (higher = focus more on hard examples)
        reduction: 'mean', 'sum', or 'none'
    
    Returns:
        focal_loss: computed focal loss
    """
    # Binary Cross Entropy with logits
    bce_loss = th.nn.functional.binary_cross_entropy_with_logits(
        pred, target, reduction='none'
    )
    
    # Get probabilities
    pred_prob = th.sigmoid(pred)
    
    # Compute focal weight
    p_t = pred_prob * target + (1 - pred_prob) * (1 - target)
    focal_weight = (1 - p_t) ** gamma
    
    # Apply alpha weighting
    alpha_t = alpha * target + (1 - alpha) * (1 - target)
    
    # Compute focal loss
    loss = alpha_t * focal_weight * bce_loss
    
    if reduction == 'mean':
        return loss.mean()
    elif reduction == 'sum':
        return loss.sum()
    else:
        return loss


def focal_loss_multiclass(pred, target, alpha=None, gamma=2.0, reduction='mean'):
    """
    Focal Loss for multi-class classification
    
    Args:
        pred: predicted logits [B, num_classes]
        target: ground truth class indices [B]
        alpha: class weights [num_classes] or None
        gamma: focusing parameter
        reduction: 'mean', 'sum', or 'none'
    
    Returns:
        focal_loss: computed focal loss
    """
    # Cross entropy loss without reduction
    ce_loss = th.nn.functional.cross_entropy(pred, target, reduction='none', weight=alpha)
    
    # Get probabilities
    pred_prob = th.softmax(pred, dim=1)
    
    # Get probability of true class
    p_t = pred_prob.gather(1, target.view(-1, 1)).squeeze(1)
    
    # Stabilize p_t to prevent numerical issues
    p_t = th.clamp(p_t, min=1e-7, max=1.0 - 1e-7)
    
    # Compute focal weight
    focal_weight = (1 - p_t) ** gamma
    
    # Compute focal loss
    loss = focal_weight * ce_loss
    
    # Clamp loss to prevent explosion
    loss = th.clamp(loss, max=10.0)
    
    if reduction == 'mean':
        return loss.mean()
    elif reduction == 'sum':
        return loss.sum()
    else:
        return loss


class UncertaintyWeightedLoss(th.nn.Module):
    """
    Uncertainty-weighted multi-task loss with learnable log_sigma parameters
    
    Loss = (1 / (2 * sigma_seg^2)) * L_seg + log(sigma_seg) 
         + (1 / (2 * sigma_cls^2)) * L_cls + log(sigma_cls)
    
    Reference: "Multi-Task Learning Using Uncertainty to Weigh Losses for Scene Geometry and Semantics"
    https://arxiv.org/abs/1705.07115
    """
    
    def __init__(self):
        super(UncertaintyWeightedLoss, self).__init__()
        # Initialize log_sigma (learnable uncertainty parameters)
        # log_sigma instead of sigma for numerical stability
        self.log_sigma_seg = th.nn.Parameter(th.zeros(1))
        self.log_sigma_cls = th.nn.Parameter(th.zeros(1))
    
    def forward(self, loss_seg, loss_cls):
        """
        Compute uncertainty-weighted loss
        
        Args:
            loss_seg: segmentation loss (scalar)
            loss_cls: classification loss (scalar)
        
        Returns:
            total_loss: weighted sum of losses
            sigma_seg: exp(log_sigma_seg) for monitoring
            sigma_cls: exp(log_sigma_cls) for monitoring
        """
        # Compute precision (1 / sigma^2) from log_sigma
        precision_seg = th.exp(-2 * self.log_sigma_seg)
        precision_cls = th.exp(-2 * self.log_sigma_cls)
        
        # Weighted losses + regularization terms
        weighted_seg = 0.5 * precision_seg * loss_seg + self.log_sigma_seg
        weighted_cls = 0.5 * precision_cls * loss_cls + self.log_sigma_cls
        
        total_loss = weighted_seg + weighted_cls
        
        # Return sigmas for monitoring
        sigma_seg = th.exp(self.log_sigma_seg)
        sigma_cls = th.exp(self.log_sigma_cls)
        
        return total_loss, sigma_seg, sigma_cls


def combined_segmentation_loss(pred, target, gamma=1.5, lambda_focal=0.5):
    """
    Combined segmentation loss: Dice + Focal
    
    Args:
        pred: predicted segmentation logits [B, C, H, W]
        target: ground truth segmentation masks [B, C, H, W]
        gamma: focal loss gamma parameter
        lambda_focal: weight for focal loss component
    
    Returns:
        combined_loss: Dice + lambda * Focal
    """
    # Dice loss
    loss_dice = dice_loss(pred, target)
    
    # Focal loss for segmentation
    loss_focal = focal_loss(pred, target, gamma=gamma)
    
    # Combined
    total_loss = loss_dice + lambda_focal * loss_focal
    
    return total_loss, loss_dice, loss_focal


# ============================================================================
