"""Keyed, reproducible negative sampler shared by every phase (explicit False + random unknown).

Each draw is seeded from (seed, sampler version, cutoff, epoch, user, positive app_id), so
every model sees exactly the same negatives regardless of batch order or model type.
"""

from collections import OrderedDict

import numpy as np

from steamrec.common import keyed_seed


def contains(sorted_values, value):
    at = np.searchsorted(sorted_values, value)
    return at < len(sorted_values) and sorted_values[at] == value


class NegativeSampler:
    def __init__(self, snapshot, config, seed):
        self.snapshot, self.config, self.seed = snapshot, config, seed
        self.candidates = snapshot.candidates[~snapshot.items["cold"][snapshot.candidates]]
        self.cache = OrderedDict()
        self.stats = {"positive_rows": 0, "explicit_draws": 0, "unknown_draws": 0,
                      "no_negative_rows": 0, "short_negative_rows": 0}
        if config["mode"] not in {"mixed", "semantic", "explicit"}:
            raise ValueError("Unknown sampler mode")
        if config["mode"] == "semantic" and snapshot.content is None:
            raise ValueError("Semantic sampler needs frozen content vectors")

    @property
    def width(self):
        return self.config["explicit_count"] + (
            0 if self.config["mode"] == "explicit" else self.config["unknown_count"])

    def semantic_pool(self, positive):
        if positive in self.cache:
            self.cache.move_to_end(positive)
            return self.cache[positive]
        content = self.snapshot.content
        query = np.asarray(content[positive])
        if not np.any(query):
            return np.empty(0, dtype=np.int64)
        # Content vectors are L2-normalized. This is content-only, never future review data.
        similarities = np.asarray(content[self.candidates]) @ query
        order = np.lexsort((self.candidates, -similarities))
        low = int(len(order) * self.config["semantic_low"])
        high = int(len(order) * self.config["semantic_high"])
        pool = self.candidates[order[low:high]]
        self.cache[positive] = pool
        if len(self.cache) > 128:
            self.cache.popitem(last=False)
        return pool

    def sample(self, user, positive, epoch):
        cfg, snap = self.config, self.snapshot
        rng = np.random.default_rng(keyed_seed(
            self.seed, cfg["version"], snap.manifest["cutoff"], epoch,
            int(user), int(snap.items["app_id"][positive])))
        reviewed = snap.row("reviewed", user)
        explicit = snap.row("explicit", user)
        explicit = explicit[snap.items["candidate"][explicit] & ~snap.items["cold"][explicit]]
        explicit_n = min(cfg["explicit_count"], len(explicit))
        negatives = list(rng.choice(explicit, explicit_n, replace=False)) if explicit_n else []
        weights = [cfg["explicit_weight"]] * explicit_n
        unknowns = []
        wanted = 0 if cfg["mode"] == "explicit" else cfg["unknown_count"]
        if wanted and cfg["mode"] == "semantic":
            pool = self.semantic_pool(positive)
            pool = pool[~np.isin(pool, reviewed)]
            number = min(int(round(wanted * cfg["semantic_fraction"])), len(pool))
            if number:
                unknowns.extend(rng.choice(pool, number, replace=False).tolist())
        # Rejection sampling: draw from the catalog, skip anything the user already reviewed.
        attempts = 0
        while len(unknowns) < wanted and len(self.candidates) and attempts < 64 * max(1, wanted):
            item = int(self.candidates[rng.integers(len(self.candidates))])
            attempts += 1
            if not contains(reviewed, item) and item not in unknowns:
                unknowns.append(item)
        if len(unknowns) < wanted:
            # User reviewed almost the whole catalog: fall back to the explicit complement.
            pool = self.candidates[~np.isin(self.candidates, reviewed)]
            pool = pool[~np.isin(pool, unknowns)]
            number = min(wanted - len(unknowns), len(pool))
            if number:
                unknowns.extend(rng.choice(pool, number, replace=False).tolist())
        negatives.extend(unknowns)
        weights.extend([cfg["unknown_weight"]] * len(unknowns))
        self.stats["positive_rows"] += 1
        self.stats["explicit_draws"] += explicit_n
        self.stats["unknown_draws"] += len(unknowns)
        self.stats["no_negative_rows"] += int(not negatives)
        self.stats["short_negative_rows"] += int(len(negatives) < self.width)
        # Pad to a fixed width; weight 0 means the padded slot never affects the loss.
        ids = np.zeros(self.width, dtype=np.int64)
        w = np.zeros(self.width, dtype=np.float32)
        ids[:len(negatives)], w[:len(weights)] = negatives, weights
        return ids, w

    def batch(self, users, positives, epoch):
        ids = np.zeros((len(users), self.width), dtype=np.int64)
        weights = np.zeros_like(ids, dtype=np.float32)
        for j, (user, positive) in enumerate(zip(users, positives)):
            ids[j], weights[j] = self.sample(int(user), int(positive), epoch)
        return ids, weights
