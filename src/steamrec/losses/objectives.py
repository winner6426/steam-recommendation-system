"""Training objectives shared by MF and Hybrid.

- weighted_bce: pointwise; label 1 for True reviews, 0 for False reviews and sampled unknowns.
  Weights let sampled unknowns (assumed negatives) count less than real False reviews.
- bpr: pairwise; -log sigmoid(s(u,p) - s(u,n)), averaged over each positive's weighted negatives.
"""

import torch
import torch.nn.functional as F


def weighted_bce(logits, labels, weights):
    if logits.numel() == 0:
        return logits.sum() * 0
    return (F.binary_cross_entropy_with_logits(logits, labels, reduction="none") * weights).mean()


def bpr(positive, negative, weights):
    """positive [B], negative [B,N], weights [B,N]; rows with no negatives are skipped."""
    valid = weights.sum(1) > 0
    if not valid.any():
        return positive.sum() * 0
    positive, negative, weights = positive[valid], negative[valid], weights[valid]
    # softplus(n - p) == -log(sigmoid(p - n)), numerically stable.
    loss = F.softplus(negative - positive[:, None]) * weights
    return (loss.sum(1) / weights.sum(1)).mean()
