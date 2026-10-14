"""Preprocess four full sources, validate, then persist four tables.

Run: python scripts/prepare_data.py --config configs/data.json
No sampling, row limit, training, or snapshot creation is performed here.
"""

import argparse
import json
import shutil
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

import duckdb

from steamrec.data.clean import clean_file
from steamrec.data.io import TABLES, connect, load_config, load_file, save_tables, write_parquet
from steamrec.data.manifest import fingerprint, write_manifest
from steamrec.data.validate import validate_schema, validate_staged


def prepare(config, root):
    root = Path(root).resolve()
    inputs = {name: (root / config["inputs"][key]).resolve() for name, key in
              zip(TABLES, ("interactions", "catalog", "users", "metadata"))}
    database = (root / config["outputs"]["database"]).resolve()
    manifest = (root / config["outputs"]["manifest"]).resolve()
    temporary = (root / config["execution"]["temporary_directory"]).resolve()
    rejected_dir = database.parent / "rejected"
    parquet_dir = (root / config["outputs"].get("parquet_directory", str(database.parent / "parquet"))).resolve()
    for path in (database, manifest, rejected_dir, parquet_dir):
        if path.exists():
            raise FileExistsError(f"Output already exists; refusing to overwrite: {path}")
    raw_root = (root / "data/raw").resolve()
    for path in (database, manifest, temporary, parquet_dir):
        if path == raw_root or raw_root in path.parents or path in inputs.values():
            raise ValueError(f"Output overlaps raw data: {path}")
    for path in inputs.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    if isinstance(config["execution"]["threads"], bool) or config["execution"]["threads"] <= 0:
        raise ValueError("threads must be positive")
    started = time.perf_counter()
    print("Fingerprinting all four inputs (full data).", flush=True)
    identities = {name: fingerprint(path) for name, path in inputs.items()}
    temporary.mkdir(parents=True, exist_ok=True)
    database.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="preprocess_", dir=temporary) as staging:
        staging = Path(staging)
        staged = {name: staging / f"{name}.parquet" for name in TABLES}
        rejected = {name: staging / f"{name}_rejected.parquet" for name in TABLES}
        with connect(config, staging) as con:
            for name in TABLES:
                print(f"Reading and normalizing all {name} rows.", flush=True)
                load_file(con, inputs[name], name)
                validate_schema(con, name)
                clean_file(con, name)
                write_parquet(con, f"{name}_ready", staged[name])
                write_parquet(con, f"{name}_rejected", rejected[name])
            print("Validating all staged data and joins before saving tables.", flush=True)
            counts, audit, monthly = validate_staged(con, staged, rejected)
        for name, path in inputs.items():
            stat = path.stat()
            if stat.st_size != identities[name]["bytes"] or stat.st_mtime_ns != identities[name]["mtime_ns"]:
                raise RuntimeError(f"Input changed during preprocessing: {path}")
        print("Saving validated tables to the persistent DuckDB database.", flush=True)
        with connect(config, staging, database) as con:
            save_tables(con, staged)
        with duckdb.connect(str(database), read_only=True) as con:
            for name in TABLES:
                actual = con.execute(f"SELECT count(*) FROM {name}").fetchone()[0]
                if actual != counts[name]["stored_rows"]:
                    raise RuntimeError(f"Persisted row count mismatch: {name}")
        rejected_dir.mkdir(exist_ok=False)
        for name in TABLES:
            rejected[name].rename(rejected_dir / f"{name}.parquet")
        # Keep complete Parquet artifacts; staging is not the final output.
        parquet_dir.mkdir(parents=True, exist_ok=False)
        filenames = {"interactions": "recommendations.parquet", "games": "games.parquet",
                     "users": "users.parquet", "content": "content.parquet"}
        parquet_artifacts = {}
        for name in TABLES:
            destination = parquet_dir / filenames[name]
            # copy2 also supports a temporary directory on another drive.
            shutil.copy2(staged[name], destination)
            with duckdb.connect() as verify:
                actual = verify.execute("SELECT count(*) FROM read_parquet(?)", [str(destination)]).fetchone()[0]
                if actual != counts[name]["stored_rows"]:
                    raise RuntimeError(f"Final Parquet row count mismatch: {name}")
            parquet_artifacts[name] = str(destination)
    report = {
        "pipeline_version": 2, "scope": "full; no LIMIT or sampling",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": round(time.perf_counter() - started, 2),
        "duckdb_version": duckdb.__version__, "config": config,
        "inputs": identities, "counts": counts, "audit": audit, "monthly": monthly,
        "artifacts": {"database": config["outputs"]["database"], "tables": list(TABLES),
                      "rejected_directory": str(rejected_dir), "parquet": parquet_artifacts},
        "rules": {"dedup": "deferred until history is limited to snapshot cutoff",
                  "invalid_rows": "quarantine with raw record and first rejection reason",
                  "orphan_reviews": "reported and retained",
                  "review_before_release": "reported and retained",
                  "content": "current metadata snapshot; no encoder fitted",
                  "columns": "preserve all 8 recommendation, 13 game and 3 user columns in specified order",
                  "features": "storage retains aggregates/prices; historical model features must exclude future aggregates"},
        "snapshot": {"T1": None, "T2": None, "seed": None, "sampler": None},
        "verified": "Source/staging counts accounted for; DB reopened; final Parquet and table counts matched; 8/13/3 schemas checked",
    }
    write_manifest(manifest, report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    report = prepare(load_config(root / args.config), root)
    print(json.dumps(report["counts"], indent=2))
    print("Manifest:", report["config"]["outputs"]["manifest"])
