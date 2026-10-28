"""Step 1: build validation (cutoff T1) and test (cutoff T2) snapshots.

Run: python scripts/prepare_snapshots.py --protocol configs/protocol.yaml --experiment main
"""

import argparse
import copy

from steamrec.common import identifier, protocol
from steamrec.data.snapshot import build_snapshot, cold_items


def experiment_config(config, scenario):
    config = copy.deepcopy(config)
    if scenario == "semantic":
        config["sampler"]["mode"] = "semantic"
    elif scenario == "explicit":
        config["sampler"]["mode"] = "explicit"
    elif scenario not in {"main", "cold"}:
        raise ValueError(scenario)
    return config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--scenario", choices=["main", "cold", "semantic", "explicit"], default="main")
    args = parser.parse_args()
    config = experiment_config(protocol(args.protocol), args.scenario)
    experiment = identifier(args.experiment)
    hidden = cold_items(config) if args.scenario == "cold" else []
    for stage in ("validation", "test"):
        print(build_snapshot(config, experiment, stage, hidden), flush=True)


if __name__ == "__main__":
    main()
