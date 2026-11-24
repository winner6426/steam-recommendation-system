"""Step 3: estimate memory/time BEFORE a full training run.

Allocates each model for real (so you see whether it fits in RAM), trains `benchmark_batches`
batches and extrapolates one epoch. Writes nothing.

Run: python scripts/estimate_resources.py --protocol configs/protocol.yaml \
         --snapshot data/splits/main/validation --trials configs/mf.yaml configs/hybrid.yaml
"""

import argparse
import json
import time

import psutil
import torch

from steamrec.common import protocol, training_seed
from steamrec.data.snapshot import Snapshot
from steamrec.training.engine import estimate, model_for, setting, train_epoch
from steamrec.training.suite import trials_from


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--snapshot", required=True)
    parser.add_argument("--trials", required=True, nargs="+")
    args = parser.parse_args()
    config = protocol(args.protocol)
    snapshot = Snapshot(args.snapshot)
    process = psutil.Process()
    print(f"available RAM: {psutil.virtual_memory().available / 2**30:.1f} GB")
    for trial in trials_from(args.trials):
        if trial["architecture"] == "hybrid" and snapshot.content is None:
            print(f"[{trial['name']}] skipped: build content first")
            continue
        static = estimate(snapshot, trial, config)
        torch.manual_seed(training_seed(config))
        model = model_for(snapshot, trial, config)
        optimizer = torch.optim.SGD(model.parameters(), lr=setting(trial, config, "learning_rate"))
        batches = config["execution"]["benchmark_batches"]
        started = time.perf_counter()
        sums = train_epoch(snapshot, model, optimizer, trial, config, epoch=1,
                           max_batches=batches, log_every=0)
        per_batch = (time.perf_counter() - started) / max(1, sums["batches"])
        epoch_minutes = per_batch * static["batches_per_epoch"] / 60
        print(f"[{trial['name']}] " + json.dumps({
            "weights_GB": round(static["weight_bytes"] / 2**30, 2),
            "process_RSS_GB": round(process.memory_info().rss / 2**30, 2),
            "batches_per_epoch": static["batches_per_epoch"],
            "seconds_per_batch": round(per_batch, 4),
            "train_minutes_per_epoch_estimate": round(epoch_minutes, 1),
            "epochs": config["execution"]["epochs"],
            "note": "excludes per-epoch full-catalog evaluation time"}))
        del model, optimizer


if __name__ == "__main__":
    main()
