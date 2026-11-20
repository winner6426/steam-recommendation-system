"""Step 4: run Phase 1 + 2 (Popularity, MF+BCE, MF+BPR, Hybrid+BCE, Hybrid+BPR).

Validation (choose configs/epochs, then lock):
  python scripts/run_experiments.py validation --protocol configs/protocol.yaml \
      --trials configs/mf.yaml configs/hybrid.yaml --run phase12 \
      --snapshot data/splits/main/validation
Test (retrain locked configs to T2, report only; run once):
  python scripts/run_experiments.py test --protocol configs/protocol.yaml \
      --trials configs/mf.yaml configs/hybrid.yaml --run phase12 \
      --snapshot data/splits/main/test --validation-snapshot data/splits/main/validation

Smoke (quick end-to-end check, never a main result): add --smoke-batches 50 --smoke-users 2000
to the validation command and use a run name such as phase12-smoke.
"""

import argparse

from steamrec.common import identifier, protocol
from steamrec.training.suite import run_test, run_validation, trials_from


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["validation", "test"])
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--trials", required=True, nargs="+")
    parser.add_argument("--run", required=True)
    parser.add_argument("--snapshot", required=True)
    parser.add_argument("--validation-snapshot")
    parser.add_argument("--smoke-batches", type=int)
    parser.add_argument("--smoke-users", type=int)
    parser.add_argument("--save-checkpoints", action="store_true",
                        help="test stage: keep the retrained models (several GB each)")
    args = parser.parse_args()
    config = protocol(args.protocol)
    trials = trials_from(args.trials)
    run = identifier(args.run)
    if args.stage == "validation":
        smoke = None
        if args.smoke_batches or args.smoke_users:
            if not (args.smoke_batches and args.smoke_users) or "smoke" not in run:
                parser.error("smoke needs both --smoke-batches and --smoke-users and a run name containing 'smoke'")
            smoke = {"batches": args.smoke_batches, "users": args.smoke_users}
        locked = run_validation(args.snapshot, config, trials, run, smoke)
        print("validation winner:", locked["winner"])
        for name, selected in locked["selected"].items():
            print(f"  {name}: epoch {selected['epoch']} {locked['selection_metric']}={selected['validation_metric']:.5f}")
    else:
        if not args.validation_snapshot:
            parser.error("test needs --validation-snapshot")
        comparison = run_test(args.snapshot, args.validation_snapshot, config, trials, run,
                              save_checkpoints=args.save_checkpoints)
        for name, values in comparison["differences"].items():
            print(name, {k: round(v, 5) for k, v in values.items() if k.startswith(("Recall", "NDCG"))})


if __name__ == "__main__":
    main()
