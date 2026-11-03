"""Phase 1 model: Matrix Factorization, s(u,i) = U[u] . V[i] / sqrt(d).

Item IDs go through `embedding_idx`: 0 is a padding row fixed at zero for games without
train reviews (unseen), so MF scores them 0 instead of a random vector. Sparse gradients +
plain SGD keep memory at one copy of the tables (no per-parameter optimizer state).
"""

import math

import numpy as np
import torch
from torch import nn


class MF(nn.Module):
    def __init__(self, snapshot, dimension):
        super().__init__()
        self.dimension = dimension
        self.users = nn.Embedding(snapshot.n_users, dimension, sparse=True)
        self.ids = nn.Embedding(snapshot.n_learned_items + 1, dimension, padding_idx=0, sparse=True)
        # item_idx (storage index of every game) -> embedding_idx (0 = never seen in train)
        self.register_buffer("item_lookup", torch.as_tensor(
            np.array(snapshot.items["embedding_idx"], copy=True), dtype=torch.long))
        nn.init.normal_(self.users.weight, std=0.05)
        nn.init.normal_(self.ids.weight, std=0.05)
        with torch.no_grad():
            self.ids.weight[0].zero_()

    def user_vectors(self, users):
        return self.users(users)

    def item_vectors(self, items):
        return self.ids(self.item_lookup[items])

    def forward(self, users, items):
        """users and items broadcast together, e.g. [B] x [B] or [B,1] x [B,N]."""
        return (self.user_vectors(users) * self.item_vectors(items)).sum(-1) / math.sqrt(self.dimension)

    def regularization(self, users, items):
        # L2 only on rows touched by this batch; never sweep all 10M+ user rows per step.
        u = self.users(torch.unique(users))
        v = self.ids(torch.unique(self.item_lookup[items.reshape(-1)]))
        return u.square().mean() + v.square().mean()
