"""Phase 2 model: Hybrid CF + content, g_i = V[i] + W c_i and s(u,i) = U[u] . g_i / sqrt(d).

Same user table and ID table as MF; adds a bias-free projection W of the frozen content
vector c_i (TF-IDF+SVD). An unseen game (embedding_idx 0) still gets W c_i, which is what
lets content help cold/sparse items. ID dropout randomly zeroes V[i] during training so
W c_i must carry signal on its own. Phase 3 reuses this class and only swaps the loss.
"""

import numpy as np
import torch
from torch import nn

from steamrec.models.matrix_factorization import MF


class Hybrid(MF):
    def __init__(self, snapshot, dimension, id_dropout=0.0):
        if snapshot.content is None:
            raise ValueError("Build content (scripts/build_content.py) before Hybrid training")
        if not 0 <= id_dropout < 1:
            raise ValueError("Invalid ID dropout")
        super().__init__(snapshot, dimension)
        self.id_dropout = id_dropout
        self.register_buffer("content", torch.as_tensor(
            np.array(snapshot.content, copy=True), dtype=torch.float32))
        self.projection = nn.Linear(self.content.shape[1], dimension, bias=False)

    def item_vectors(self, items):
        vectors = super().item_vectors(items)
        if self.training and self.id_dropout:
            keep = torch.rand(vectors.shape[:-1], device=vectors.device) >= self.id_dropout
            vectors = vectors * keep.unsqueeze(-1)
        return vectors + self.projection(self.content[items])

    def regularization(self, users, items):
        return super().regularization(users, items) + self.projection.weight.square().mean()
