"""Clean the game catalog and audit a supplied interaction Parquet file.

Run from the project root: python scripts/prepare_catalog.py --config configs/catalog.json
Paths in the config are relative to the project root, or may be absolute.
"""

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import duckdb


def sql_path(path):
    return "'" + path.as_posix().replace("'", "''") + "'"


def fingerprint(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return {"bytes": path.stat().st_size, "sha256": digest.hexdigest()}


def prepare(config, root):
    paths = {key: (root / config[key]).resolve() for key in (
        "games_csv", "interactions_parquet", "output_dir", "manifest"
    )}
    for key in ("games_csv", "interactions_parquet"):
        if not paths[key].is_file():
            raise FileNotFoundError(paths[key])
    # Each run gets a new directory; never overwrite an existing artifact.
    if paths["output_dir"].exists() or paths["manifest"].exists():
        raise FileExistsError("Output or manifest exists; choose a new run in config.")

    with duckdb.connect() as con:
        con.execute("SET memory_limit = ?", [config["memory_limit"]])
        con.execute("SET threads = ?", [config["threads"]])
        con.execute("""
            CREATE TEMP TABLE raw_games AS
            SELECT * FROM read_csv(?, header=true, all_varchar=true)
        """, [str(paths["games_csv"])])
        con.execute("""
            CREATE TEMP TABLE games AS
            SELECT
                CASE WHEN regexp_full_match(trim(app_id), '[0-9]+')
                     THEN TRY_CAST(trim(app_id) AS BIGINT) END AS app_id,
                NULLIF(trim(title), '') AS title,
                CASE WHEN regexp_full_match(trim(date_release), '[0-9]{4}-[0-9]{2}-[0-9]{2}')
                     THEN CAST(try_strptime(trim(date_release), '%Y-%m-%d') AS DATE)
                     END AS date_release,
                CASE lower(trim(win)) WHEN 'true' THEN true WHEN 'false' THEN false END AS win,
                CASE lower(trim(mac)) WHEN 'true' THEN true WHEN 'false' THEN false END AS mac,
                CASE lower(trim(linux)) WHEN 'true' THEN true WHEN 'false' THEN false END AS linux,
                app_id AS raw_app_id,
                date_release AS raw_date_release
            FROM raw_games
        """)
        duplicates = con.execute("""
            SELECT app_id, count(*) AS rows FROM games
            WHERE app_id IS NOT NULL AND app_id > 0
            GROUP BY app_id HAVING count(*) > 1 ORDER BY app_id
        """).fetchall()
        if duplicates:
            raise ValueError(
                f"Duplicate catalog IDs ({len(duplicates)} groups): {duplicates[:10]}. "
                "Resolve the source conflict explicitly; no automatic dedup."
            )
        con.execute("""
            CREATE TEMP VIEW clean_games AS
            SELECT app_id, title, date_release, win, mac, linux,
                   date_release IS NULL AS missing_release_date,
                   title IS NULL AS missing_title
            FROM games WHERE app_id IS NOT NULL AND app_id > 0
        """)
        con.execute("""
            CREATE TEMP VIEW rejected_games AS
            SELECT *, 'invalid_app_id' AS rejection_reason
            FROM games WHERE app_id IS NULL OR app_id <= 0
        """)
        con.execute("CREATE TEMP VIEW interactions AS SELECT * FROM read_parquet(" +
                    sql_path(paths["interactions_parquet"]) + ")")
        # Audit only: never delete or relabel reviews based on this join.
        con.execute("""
            CREATE TEMP VIEW orphan_reviews AS
            SELECT i.* FROM interactions i
            WHERE NOT EXISTS (SELECT 1 FROM clean_games g WHERE g.app_id = i.app_id)
        """)
        counts = {
            "catalog_input_rows": con.execute("SELECT count(*) FROM games").fetchone()[0],
            "catalog_clean_rows": con.execute("SELECT count(*) FROM clean_games").fetchone()[0],
            "catalog_rejected_rows": con.execute("SELECT count(*) FROM rejected_games").fetchone()[0],
            "missing_release_date": con.execute("SELECT count(*) FROM clean_games WHERE missing_release_date").fetchone()[0],
            "missing_title": con.execute("SELECT count(*) FROM clean_games WHERE missing_title").fetchone()[0],
            "duplicate_app_id_groups": len(duplicates),
            "interaction_rows_checked": con.execute("SELECT count(*) FROM interactions").fetchone()[0],
            "orphan_review_rows": con.execute("SELECT count(*) FROM orphan_reviews").fetchone()[0],
            "orphan_game_ids": con.execute("SELECT count(DISTINCT app_id) FROM orphan_reviews").fetchone()[0],
            "reviews_with_missing_release_date": con.execute("""
                SELECT count(*) FROM interactions i JOIN clean_games g USING (app_id)
                WHERE g.missing_release_date
            """).fetchone()[0],
        }
        if counts["catalog_input_rows"] != counts["catalog_clean_rows"] + counts["catalog_rejected_rows"]:
            raise RuntimeError("Catalog row accounting failed")

        paths["output_dir"].mkdir(parents=True, exist_ok=False)
        outputs = {}
        for view, filename, expected in (
            ("clean_games", "games_clean.parquet", counts["catalog_clean_rows"]),
            ("rejected_games", "games_rejected.parquet", counts["catalog_rejected_rows"]),
            ("orphan_reviews", "orphan_reviews.parquet", counts["orphan_review_rows"]),
        ):
            target = paths["output_dir"] / filename
            con.execute(f"COPY (SELECT * FROM {view}) TO {sql_path(target)} (FORMAT PARQUET, COMPRESSION ZSTD)")
            written = con.execute("SELECT count(*) FROM read_parquet(?)", [str(target)]).fetchone()[0]
            if written != expected:
                raise RuntimeError(f"Parquet row accounting failed: {target}")
            outputs[filename] = fingerprint(target)

    report = {
        "pipeline_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "duckdb_version": duckdb.__version__,
        "config": config,
        "rules": {
            "app_id": "digits only, positive BIGINT; invalid rows quarantined",
            "duplicates": "fail before output; no automatic dedup",
            "release_date": "strict YYYY-MM-DD; invalid/missing becomes NULL; retain game with flag",
            "platform": "trim/lower true/false; other values become NULL",
            "features": "title/release/platform only; exclude review aggregates and prices",
            "join": "report missing catalog IDs; preserve input reviews",
        },
        "snapshot": {"cutoff": None, "seed": None, "sampler": None,
                     "note": "Preparation only; no split, sampling or ID mapping fitted."},
        "inputs": {key: fingerprint(paths[key]) for key in ("games_csv", "interactions_parquet")},
        "counts": counts,
        "artifacts": outputs,
        "verified": "All output Parquet row counts read back and matched.",
    }
    paths["manifest"].parent.mkdir(parents=True, exist_ok=True)
    with paths["manifest"].open("x", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text(encoding="utf-8-sig"))
    report = prepare(config, Path(__file__).resolve().parents[1])
    print(json.dumps(report["counts"], indent=2))
    print("Manifest:", config["manifest"])
