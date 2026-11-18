"""Train loop shared by MF and Hybrid (Phase 1 and 2).

One epoch = one pass over EVERY retained train review of the snapshot, in review_id order,
in batches of `batch_size`. Per batch:
  BCE: score all True/False reviews in the batch + sampled unknown negatives of each positive.
  BPR: pair each True review with its sampled negatives (explicit False + unknown).
Then backprop, clip, SGD step. After each epoch (validation stage) the full-catalog evaluator
runs on all eligible users so the best epoch can be chosen.
"""

import math
import time
from pathlib import Path
import uuid

import numpy as np
import torch

from steamrec.common import digest, output_directory, record, runtime, training_seed
from steamrec.data.sampling import NegativeSampler
from steamrec.evaluation.ranking import evaluate
from steamrec.losses.objectives import bpr, weighted_bce
from steamrec.models.hybrid import Hybrid
from steamrec.models.matrix_factorization import MF

OBJECTIVES = {"bce", "bpr"}


def setting(trial, config, key):
    """A trial may override an execution hyperparameter (e.g. learning_rate); else protocol value."""
    return trial.get(key, config["execution"][key])


def model_for(snapshot, trial, config):
    dimension = setting(trial, config, "embedding_dimension")
    if trial["architecture"] == "mf":
        model = MF(snapshot, dimension)
    elif trial["architecture"] == "hybrid":
        model = Hybrid(snapshot, dimension, trial.get("id_dropout", 0.0))
    else:
        raise ValueError("Unknown model architecture")
    if trial["objective"] not in OBJECTIVES:
        raise ValueError(f"Objective not implemented for Phase 1-2: {trial['objective']}")
    model.objective = trial["objective"]
    return model


def training_identity(snapshot, trial, config):
    """Hash of everything that determines a training run: data snapshot, content, trial, protocol."""
    return digest({"snapshot": snapshot.manifest, "content": snapshot.content_manifest,
                   "trial": trial, "protocol": config})


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)


def clip_gradients(parameters, limit):
    # torch.nn.utils.clip_grad_norm_ does not support sparse embedding gradients.
    if limit <= 0:
        return
    gradients = []
    for parameter in parameters:
        if parameter.grad is None:
            continue
        if parameter.grad.is_sparse:
            parameter.grad = parameter.grad.coalesce()
            gradients.append(parameter.grad.values())
        else:
            gradients.append(parameter.grad)
    if not gradients:
        return
    squared = sum(g.detach().float().square().sum() for g in gradients)
    factor = (limit / (squared.sqrt() + 1e-12)).clamp(max=1)
    for gradient in gradients:
        gradient.mul_(factor)


def step_loss(model, batch, sampler, trial, config, epoch):
    device = config["execution"]["device"]
    s = config["sampler"]
    # Arrow-backed arrays may be read-only; copy before handing them to torch.
    labels_np = np.array(batch["label"], dtype=bool)
    all_users_np = np.array(batch["user_idx"], dtype=np.int64)
    all_items_np = np.array(batch["item_idx"], dtype=np.int64)
    pos_users_np = all_users_np[labels_np]
    pos_items_np = all_items_np[labels_np]
    negative_ids, negative_weights = sampler.batch(pos_users_np, pos_items_np, epoch)
    n = torch.as_tensor(negative_ids, dtype=torch.long, device=device)
    w = torch.as_tensor(negative_weights, dtype=torch.float32, device=device)
    pu = torch.as_tensor(pos_users_np, dtype=torch.long, device=device)
    pi = torch.as_tensor(pos_items_np, dtype=torch.long, device=device)
    if trial["objective"] == "bce":
        users = torch.as_tensor(all_users_np, dtype=torch.long, device=device)
        items = torch.as_tensor(all_items_np, dtype=torch.long, device=device)
        labels = torch.as_tensor(labels_np, dtype=torch.float32, device=device)
        logits = model(users, items)
        # True reviews weight 1; real False reviews weight explicit_weight.
        weights = torch.where(labels > 0, torch.ones_like(labels),
                              torch.full_like(labels, s["explicit_weight"]))
        # False reviews are already in this batch as rows, so only ADD the sampled unknowns.
        unknown = w == s["unknown_weight"]
        touched_users, touched_items = [users], [items]
        if unknown.any():
            unknown_users = pu[:, None].expand_as(n)[unknown]
            unknown_items = n[unknown]
            logits = torch.cat((logits, model(unknown_users, unknown_items)))
            labels = torch.cat((labels, torch.zeros_like(w[unknown])))
            weights = torch.cat((weights, w[unknown]))
            touched_users.append(unknown_users)
            touched_items.append(unknown_items)
        loss = weighted_bce(logits, labels, weights)
        with torch.no_grad():
            pos = model(pu, pi) if len(pu) else None
            neg = model(pu[:, None].expand_as(n), n) if len(pu) else None
    else:
        if len(pu) == 0 or not (w > 0).any():
            return None, {"pairs": 0, "positive_above_negative": 0}
        pos = model(pu, pi)
        neg = model(pu[:, None].expand_as(n), n)
        loss = bpr(pos, neg, w)
        touched_users, touched_items = [pu], [pi, n[w > 0]]
    diagnostics = {"pairs": 0, "positive_above_negative": 0}
    if pos is not None:
        live = w > 0
        diagnostics = {"pairs": int(live.sum()),
                       "positive_above_negative": int(((pos.detach()[:, None] > neg.detach()) & live).sum())}
    l2 = setting(trial, config, "l2")
    loss = loss + l2 * model.regularization(torch.cat(touched_users),
                                            torch.cat([x.reshape(-1) for x in touched_items]))
    return loss, diagnostics


def train_epoch(snapshot, model, optimizer, trial, config, epoch, max_batches=None, log_every=2000):
    model.train()
    sampler = NegativeSampler(snapshot, config["sampler"], training_seed(config))
    sums = {"loss_sum": 0.0, "steps": 0, "review_rows": 0, "positive_rows": 0, "false_rows": 0,
            "batches": 0, "pairs": 0, "positive_above_negative": 0, "seconds": 0.0}
    started = time.perf_counter()
    last_review = None
    for batch in snapshot.train_batches(setting(trial, config, "batch_size")):
        review_ids = batch["review_id"]
        if np.any(np.diff(review_ids) <= 0) or (last_review is not None and review_ids[0] <= last_review):
            raise ValueError("Train reviews duplicated or out of order")
        last_review = int(review_ids[-1])
        optimizer.zero_grad(set_to_none=True)
        loss, diagnostics = step_loss(model, batch, sampler, trial, config, epoch)
        if loss is not None:
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite training loss")
            loss.backward()
            clip_gradients(model.parameters(), config["execution"]["max_grad_norm"])
            optimizer.step()
            sums["loss_sum"] += float(loss.detach())
            sums["steps"] += 1
        positives = int(np.count_nonzero(batch["label"]))
        sums["review_rows"] += len(review_ids)
        sums["positive_rows"] += positives
        sums["false_rows"] += len(review_ids) - positives
        sums["batches"] += 1
        for key, value in diagnostics.items():
            sums[key] += value
        if log_every and sums["batches"] % log_every == 0:
            elapsed = time.perf_counter() - started
            print(f"  [{trial['name']}] epoch {epoch} batch {sums['batches']} "
                  f"loss {sums['loss_sum'] / max(1, sums['steps']):.4f} {elapsed:.0f}s", flush=True)
        if max_batches is not None and sums["batches"] >= max_batches:
            break
    sums["seconds"] = time.perf_counter() - started
    sums["sampler"] = dict(sampler.stats)
    sums["mean_loss"] = sums["loss_sum"] / max(1, sums["steps"])
    sums["positive_above_negative_rate"] = sums["positive_above_negative"] / max(1, sums["pairs"])
    return sums


def check_full_pass(snapshot, sums):
    stats = snapshot.manifest["stats"]
    if (sums["review_rows"] != stats["train_rows"] or sums["positive_rows"] != stats["train_positive_rows"]
            or sums["false_rows"] != stats["train_false_rows"]):
        raise ValueError("Training did not visit every retained True/False review")


def fit(snapshot, trial, config, report_dir, epochs, validate=True, checkpoint=None, smoke=None):
    """Train `epochs` full passes; write one small JSON report per epoch.

    smoke = {"batches": N, "users": M} limits work for a quick check; such reports are
    marked scope "smoke" and must never be presented as main results.
    Returns (model, reports).
    """
    if not isinstance(epochs, int) or isinstance(epochs, bool) or epochs <= 0:
        raise ValueError("epochs must be a positive integer")
    report_dir = output_directory(report_dir)
    set_seed(training_seed(config))
    device = config["execution"]["device"]
    model = model_for(snapshot, trial, config).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=setting(trial, config, "learning_rate"))
    identity = training_identity(snapshot, trial, config)
    reports = []
    for epoch in range(1, epochs + 1):
        sums = train_epoch(snapshot, model, optimizer, trial, config, epoch,
                           max_batches=smoke["batches"] if smoke else None)
        if smoke is None:
            check_full_pass(snapshot, sums)
        evaluation = (evaluate(snapshot, model, config, max_users=smoke["users"] if smoke else None)
                      if validate else None)
        report = {"epoch": epoch, "trial": trial, "identity": identity, "seed": training_seed(config),
                  "scope": "smoke" if smoke else snapshot.manifest["scope"],
                  "stage": snapshot.manifest["stage"], "snapshot_hash": digest(snapshot.manifest),
                  "training": sums, "evaluation": evaluation, "runtime": runtime()}
        record(report_dir / f"epoch-{epoch:03d}.json", report)
        reports.append(report)
        metric = config["evaluation"]["selection_metric"]
        shown = f" {metric} {evaluation['groups']['all'][metric]:.5f}" if evaluation else ""
        print(f"[{trial['name']}] epoch {epoch}/{epochs} loss {sums['mean_loss']:.4f} "
              f"pos>neg {sums['positive_above_negative_rate']:.3f}{shown}", flush=True)
    if checkpoint is not None:
        save_model(checkpoint, model, identity, trial, epochs)
    return model, reports


def save_model(path, model, identity, trial, epochs):
    path = Path(path)
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {"version": 1, "identity": identity, "trial": trial, "epochs": epochs,
             "model": model.state_dict()}
    partial = path.with_name(path.name + ".partial-" + uuid.uuid4().hex)
    with partial.open("xb") as handle:
        torch.save(state, handle)
    partial.rename(path)


def load_model(snapshot, trial, config, path):
    """Only load checkpoints this pipeline wrote (torch.load unpickles)."""
    model = model_for(snapshot, trial, config).to(config["execution"]["device"])
    state = torch.load(path, map_location="cpu", weights_only=False)
    if state["identity"] != training_identity(snapshot, trial, config):
        raise ValueError("Checkpoint identity mismatch")
    model.load_state_dict(state["model"])
    model.eval()
    return model


def estimate(snapshot, trial, config):
    """Static lower bound on memory; measure real time/RSS with scripts/estimate_resources.py."""
    dimension = setting(trial, config, "embedding_dimension")
    parameters = (snapshot.n_users + snapshot.n_learned_items + 1) * dimension
    content_bytes = 0
    if trial["architecture"] == "hybrid":
        if snapshot.content is None:
            raise ValueError("Build content before estimating Hybrid")
        parameters += dimension * snapshot.content.shape[1]
        content_bytes = snapshot.content.nbytes
    batches = math.ceil(snapshot.manifest["stats"]["train_rows"] / setting(trial, config, "batch_size"))
    return {"parameters": parameters, "weight_bytes": parameters * 4,
            "frozen_content_bytes": content_bytes, "batches_per_epoch": batches,
            "checkpoint_bytes_lower_bound": parameters * 4,
            "optimizer": "SGD without momentum (no extra state); touched-row L2"}
