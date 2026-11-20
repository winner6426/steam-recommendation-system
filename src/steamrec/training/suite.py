"""Orchestrate Phase 1-2: validation (choose) -> lock -> test (report only).

Validation (snapshot at T1): evaluate Popularity, train every trial, evaluate after each epoch,
keep the best (trial, epoch) per model family by `selection_metric`, write locked.json.
Test (snapshot at T2): refuse if config/trials/validation snapshot changed since locking,
retrain each locked family from scratch for its locked epoch count, evaluate once, write
test.csv + comparison.json (Hybrid minus MF differences). Test never re-selects anything.
"""

import csv
import io
from pathlib import Path

from steamrec.common import (base_protocol, digest, identifier, output_directory, read_json,
                             read_yaml, record)
from steamrec.data.snapshot import Snapshot
from steamrec.evaluation.ranking import evaluate
from steamrec.training.engine import OBJECTIVES, fit

REQUIRED_FAMILIES = {"mf+bce", "mf+bpr", "hybrid+bce", "hybrid+bpr"}
CONTRASTS = (("mf+bce", "hybrid+bce"), ("mf+bpr", "hybrid+bpr"))


def family(trial):
    return trial["architecture"] + "+" + trial["objective"]


def trials_from(paths):
    trials = []
    for path in paths:
        cfg = read_yaml(path)
        for entry in cfg["trials"]:
            trial = {"architecture": cfg["architecture"], **entry}
            identifier(trial["name"])
            if trial["objective"] not in OBJECTIVES:
                raise ValueError(f"Unknown objective: {trial['objective']}")
            trials.append(trial)
    if len({t["name"] for t in trials}) != len(trials):
        raise ValueError("Trial names must be unique")
    missing = REQUIRED_FAMILIES - {family(t) for t in trials}
    if missing:
        raise ValueError(f"Suite is missing required families (Hybrid+BPR is mandatory): {sorted(missing)}")
    return trials


def write_table(path, rows):
    metric_names = sorted({k for r in rows for k in r["evaluation"]["groups"]["all"] if "@" in k})
    columns = (["model", "trial", "epoch", "scope", "users", "targets"] + metric_names
               + ["sparse_user_NDCG@10", "natural_unseen_item_Recall@10", "evaluation_seconds"])
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=columns)
    writer.writeheader()
    for row in rows:
        ev, groups = row["evaluation"], row["evaluation"]["groups"]
        values = {"model": row["model"], "trial": row["trial"]["name"] if row["trial"] else "popularity",
                  "epoch": row["epoch"], "scope": ev["scope"], **groups["all"],
                  "sparse_user_NDCG@10": groups.get("sparse_user", {}).get("NDCG@10", ""),
                  "natural_unseen_item_Recall@10": groups.get("natural_unseen_item", {}).get("Recall@10", ""),
                  "evaluation_seconds": round(ev["seconds"], 2)}
        writer.writerow({key: values.get(key, "") for key in columns})
    with Path(path).open("x", encoding="utf-8", newline="") as handle:
        handle.write(buffer.getvalue())


def check_snapshot(snapshot, stage, config, trials):
    if snapshot.manifest["stage"] != stage:
        raise ValueError(f"Expected a {stage} snapshot")
    if snapshot.manifest["protocol_hash"] != digest(base_protocol(config)):
        raise ValueError("Snapshot was built with a different protocol config")
    if any(t["architecture"] == "hybrid" for t in trials) and snapshot.content is None:
        raise ValueError("Hybrid trials need content: run scripts/build_content.py first")


def run_validation(snapshot_dir, config, trials, run, smoke=None):
    snapshot = Snapshot(snapshot_dir)
    check_snapshot(snapshot, "validation", config, trials)
    reports = output_directory(Path(config["paths"]["reports"]) / run)
    metric = config["evaluation"]["selection_metric"]
    max_users = smoke["users"] if smoke else None
    popularity = {"model": "popularity", "trial": None, "epoch": 0,
                  "evaluation": evaluate(snapshot, None, config, max_users=max_users)}
    record(reports / "popularity-validation.json", popularity)
    print(f"[popularity] {metric} {popularity['evaluation']['groups']['all'][metric]:.5f}", flush=True)
    selections = {"popularity": popularity}
    for trial in trials:
        _, entries = fit(snapshot, trial, config, reports / "validation" / trial["name"],
                         config["execution"]["epochs"], validate=True, smoke=smoke)
        for entry in entries:
            row = {"model": family(trial), "trial": trial, "epoch": entry["epoch"],
                   "evaluation": entry["evaluation"]}
            best = selections.get(row["model"])
            if best is None or row["evaluation"]["groups"]["all"][metric] > best["evaluation"]["groups"]["all"][metric]:
                selections[row["model"]] = row
    rows = list(selections.values())
    write_table(reports / "validation.csv", rows)
    winner = max(rows, key=lambda r: r["evaluation"]["groups"]["all"][metric])["model"]
    locked = {"version": 1, "run": run, "scope": "smoke" if smoke else snapshot.manifest["scope"],
              "smoke": smoke, "config": config, "config_hash": digest(config),
              "trials": trials, "trials_hash": digest(trials), "selection_metric": metric,
              "winner": winner,
              "selected": {name: {"trial": r["trial"], "epoch": r["epoch"],
                                  "validation_metric": r["evaluation"]["groups"]["all"][metric]}
                           for name, r in selections.items()},
              "validation_snapshot_hash": digest(snapshot.manifest),
              "validation_content": snapshot.content_manifest}
    record(reports / "locked.json", locked)
    snapshot.close()
    return locked


def run_test(snapshot_dir, validation_dir, config, trials, run, save_checkpoints=False):
    reports = Path(config["paths"]["reports"]) / run
    locked = read_json(reports / "locked.json")
    validation = Snapshot(validation_dir)
    snapshot = Snapshot(snapshot_dir)
    if locked["config_hash"] != digest(config) or locked["trials_hash"] != digest(trials):
        raise ValueError("Config/trials changed after validation selection")
    if locked["validation_snapshot_hash"] != digest(validation.manifest):
        raise ValueError("Validation snapshot changed after selection")
    if locked["validation_content"] != validation.content_manifest:
        raise ValueError("Validation content changed after selection")
    check_snapshot(snapshot, "test", config, trials)
    if (snapshot.manifest["preprocess_identity"] != validation.manifest["preprocess_identity"]
            or snapshot.manifest["cold_app_ids"] != validation.manifest["cold_app_ids"]):
        raise ValueError("Validation/test snapshots come from different sources or cold sets")
    validation.close()
    if (reports / "comparison.json").exists():
        raise FileExistsError("Test already reported for this run; never repeat or reselect")
    test_dir = output_directory(reports / "test")
    smoke = locked["smoke"]
    max_users = smoke["users"] if smoke else None
    rows = []
    for name, selected in locked["selected"].items():
        trial, epochs = selected["trial"], selected["epoch"]
        if trial is None:
            model = None
        else:
            checkpoint = (Path(config["paths"]["checkpoints"]) / run / "test" / f"{trial['name']}.pt"
                          if save_checkpoints else None)
            # Retrain from scratch on history <= T2 with the locked config and epoch count.
            model, _ = fit(snapshot, trial, config, test_dir / trial["name"], epochs,
                           validate=False, checkpoint=checkpoint, smoke=smoke)
        row = {"model": name, "trial": trial, "epoch": epochs,
               "evaluation": evaluate(snapshot, model, config, max_users=max_users)}
        record(test_dir / f"{name.replace('+', '-')}.json", row)
        rows.append(row)
        del model
    write_table(reports / "test.csv", rows)
    metrics = {row["model"]: row["evaluation"]["groups"]["all"] for row in rows}
    differences = {f"{after} minus {before}": {k: metrics[after][k] - metrics[before][k]
                                               for k in metrics[before] if "@" in k}
                   for before, after in CONTRASTS}
    comparison = {"version": 1, "run": run, "scope": locked["scope"],
                  "locked_identity": digest(locked), "test_snapshot_hash": digest(snapshot.manifest),
                  "rows": rows, "differences": differences,
                  "winner_from_validation": locked["winner"],
                  "notes": "Test only reports locked configurations; it never selects a winner."}
    record(reports / "comparison.json", comparison)
    snapshot.close()
    return comparison
