"""Full-catalog Top-K evaluation shared by Popularity and all trained models.

For each eligible user: score every candidate game at the cutoff, mask games the user
already reviewed, keep Top-K with a deterministic tie-break (lower item_idx first), then
compare with the future positives (targets). No sampled candidates, no dense user x item matrix
beyond one user batch.
"""

import math
import time

import numpy as np
import pyarrow.parquet as pq


def topk(scores, items, k):
    """Bounded top-k with deterministic item-ID tie handling (used by tests and small inputs)."""
    scores = np.asarray(scores, dtype=np.float64)
    scores = np.where(np.isnan(scores), -np.inf, scores)
    items = np.asarray(items, dtype=np.int64)
    valid = np.flatnonzero(np.isfinite(scores))
    if len(valid) > k:
        threshold = np.partition(scores[valid], len(valid) - k)[len(valid) - k]
        above = valid[scores[valid] > threshold]
        ties = valid[scores[valid] == threshold]
        ties = ties[np.argsort(items[ties], kind="stable")[:k - len(above)]]
        valid = np.concatenate((above, ties))
    order = np.lexsort((items[valid], -scores[valid]))
    return scores[valid][order][:k], items[valid][order][:k]


def row_topk(scores, k):
    """Top-k positions of one score row; ties go to the lower position (= lower item_idx)."""
    finite = np.isfinite(scores)
    if finite.sum() <= k:
        chosen = np.flatnonzero(finite)
    else:
        threshold = np.partition(scores, len(scores) - k)[len(scores) - k]
        above = np.flatnonzero(scores > threshold)
        ties = np.flatnonzero(scores == threshold)[:k - len(above)]
        chosen = np.concatenate((above, ties))
    return chosen[np.lexsort((chosen, -scores[chosen]))]


def metrics(ranking, targets, k):
    targets = set(int(i) for i in targets)
    if not targets:
        raise ValueError("Metrics require at least one valid target")
    hits = np.asarray([int(i) in targets for i in ranking[:k]], dtype=np.float64)
    dcg = (hits / np.log2(np.arange(2, len(hits) + 2))).sum()
    ideal = (1 / np.log2(np.arange(2, min(k, len(targets)) + 2))).sum()
    return {f"Recall@{k}": float(hits.sum() / len(targets)),
            f"NDCG@{k}": float(dcg / ideal), f"HitRate@{k}": float(hits.any())}


def scorer(snapshot, model, device):
    """Return f(users) -> float32 scores [len(users), len(candidates)].

    Item vectors are computed once per evaluation (model in eval mode: no ID dropout).
    """
    candidates = snapshot.candidates
    if model is None:
        popularity = snapshot.items["popularity"][candidates].astype(np.float32)
        return lambda users: np.broadcast_to(popularity, (len(users), len(candidates))).copy()
    import torch
    model.eval()
    with torch.no_grad():
        item_vectors = model.item_vectors(torch.as_tensor(candidates, dtype=torch.long, device=device))
    scale = math.sqrt(model.dimension)

    def score(users):
        with torch.no_grad():
            u = model.user_vectors(torch.as_tensor(users, dtype=torch.long, device=device))
            return (u @ item_vectors.T / scale).float().cpu().numpy()
    return score


def evaluate(snapshot, model, config, max_users=None):
    """Evaluate every eligible user. `max_users` is for smoke runs only and marks the report."""
    started = time.perf_counter()
    ks = config["evaluation"]["ks"]
    k_max = max(ks)
    sparse_user = config["evaluation"]["sparse_user_max"]
    sparse_item = config["evaluation"]["sparse_item_max"]
    counts = pq.read_table(snapshot.root / "users.parquet", columns=["review_count"]).column(0).to_numpy()
    candidates = snapshot.candidates
    # item_idx -> column position in the score matrix (-1 if not a candidate at this cutoff).
    position = np.full(snapshot.n_items, -1, dtype=np.int64)
    position[candidates] = np.arange(len(candidates))
    score = scorer(snapshot, model, config["execution"]["device"])
    totals, numbers, target_numbers = {}, {}, {}
    coverage = {k: set() for k in ks}
    examples = []

    def add(group, ranked, targets):
        if len(targets) == 0:
            return
        numbers[group] = numbers.get(group, 0) + 1
        target_numbers[group] = target_numbers.get(group, 0) + len(targets)
        sums = totals.setdefault(group, {})
        for k in ks:
            for key, value in metrics(ranked, targets, k).items():
                sums[key] = sums.get(key, 0.0) + value

    evaluated = 0
    for users in snapshot.eligible_users(config["execution"]["user_batch_size"]):
        if len(users) == 0:
            continue
        if max_users is not None:
            users = users[:max_users - evaluated]
            if len(users) == 0:
                break
        scores = score(users)
        for row, user in enumerate(users):
            user = int(user)
            reviewed = position[snapshot.row("reviewed", user)]
            scores[row, reviewed[reviewed >= 0]] = -np.inf
            ranked = candidates[row_topk(scores[row], k_max)]
            targets = snapshot.row("targets", user)
            add("all", ranked, targets)
            add("sparse_user" if counts[user] <= sparse_user else "warm_user", ranked, targets)
            train_counts = snapshot.items["train_count"][targets]
            add("sparse_item", ranked, targets[(train_counts > 0) & (train_counts <= sparse_item)])
            add("warm_item", ranked, targets[train_counts > sparse_item])
            add("natural_unseen_item", ranked,
                targets[(train_counts == 0) & ~snapshot.items["cold"][targets]])
            add("simulated_cold_item", ranked, targets[snapshot.items["cold"][targets]])
            for k in ks:
                coverage[k].update(int(i) for i in ranked[:k])
            if len(examples) < 5:
                examples.append({"user_idx": user,
                                 "app_ids": snapshot.items["app_id"][ranked].tolist()})
        evaluated += len(users)
    eligible = snapshot.manifest["stats"]["eligible_users"]
    complete = max_users is None
    if eligible == 0 or numbers.get("all", 0) == 0:
        raise ValueError("Snapshot has no eligible users")
    if complete and numbers["all"] != eligible:
        raise ValueError("Evaluator did not process every eligible user")
    groups = {group: {**{k: v / numbers[group] for k, v in sums.items()},
                      "users": numbers[group], "targets": target_numbers[group]}
              for group, sums in totals.items()}
    groups["all"].update({f"Coverage@{k}": len(coverage[k]) / len(candidates) for k in ks})
    return {"scope": snapshot.manifest["scope"] if complete else "smoke",
            "ranking": "full-catalog",
            "evaluation_coverage": {"complete": complete, "expected_users": eligible,
                                    "evaluated_users": numbers["all"],
                                    "candidate_items_per_user": len(candidates),
                                    "sampled_candidates": False},
            "stage": snapshot.manifest["stage"], "cutoff": snapshot.manifest["cutoff"],
            "groups": groups, "target_audit": snapshot.manifest["stats"]["targets_by_reason"],
            "seconds": time.perf_counter() - started, "examples": examples}
