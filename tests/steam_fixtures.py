"""Tiny DuckDB + protocol fixtures shared by the snapshot and Phase 1-2 tests."""

import json
from pathlib import Path

import duckdb
import yaml

from steamrec.common import protocol


def make_database(root, reviews, games, texts=None):
    """reviews: (user_id, app_id, date, is_recommended, review_id); games: (app_id, release)."""
    root = Path(root)
    database = root / "steamrec.duckdb"
    texts = texts or {}
    with duckdb.connect(str(database)) as con:
        con.execute("""CREATE TABLE interactions(app_id BIGINT, helpful BIGINT, funny BIGINT,
            date DATE, is_recommended BOOLEAN, hours DOUBLE, user_id BIGINT, review_id BIGINT)""")
        con.executemany("INSERT INTO interactions VALUES (?,0,0,?,?,1.0,?,?)",
                        [(a, d, r, u, i) for u, a, d, r, i in reviews])
        con.execute("CREATE TABLE games(app_id BIGINT, title VARCHAR, date_release DATE)")
        con.executemany("INSERT INTO games VALUES (?, 'g', ?)", games)
        con.execute("CREATE TABLE users AS SELECT DISTINCT user_id, 1 AS products, 1 AS reviews FROM interactions")
        con.execute("CREATE TABLE content(app_id BIGINT, text VARCHAR, missing_content BOOLEAN)")
        con.executemany("INSERT INTO content VALUES (?, ?, ?)",
                        [(a, texts.get(a, ""), not texts.get(a)) for a, _ in games])
        counts = {name: con.execute(f"SELECT count(*) FROM {name}").fetchone()[0]
                  for name in ("interactions", "games", "users", "content")}
    pre = root / "preprocess.json"
    pre.write_text(json.dumps({
        "scope": "full; no LIMIT or sampling", "inputs": {},
        "artifacts": {"database": str(database)},
        "counts": {k: {"stored_rows": v} for k, v in counts.items()}}), encoding="utf-8")
    return database, pre


def make_config(root, database, pre, **execution):
    root = Path(root)
    config = yaml.safe_load(Path("configs/protocol.yaml").read_text(encoding="utf-8"))
    config.update({"T1": "2020-06-30", "T2": "2020-09-30"})
    config["paths"].update({
        "database": str(database), "preprocess_manifest": str(pre),
        "splits": str(root / "splits"), "manifests": str(root / "manifests"),
        "reports": str(root / "runs"), "checkpoints": str(root / "checkpoints")})
    config["execution"].update({"temporary_directory": str(root / "tmp"), "memory_limit": "256MB",
                                "threads": 1, "user_batch_size": 2, **execution})
    config["content"].update({"dimensions": 8, "min_df": 1})
    path = root / "protocol.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return protocol(path), path
