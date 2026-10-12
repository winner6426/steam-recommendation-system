"""Read-only full CSV audit; write a small JSON report without changing inputs."""

import argparse
import json
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

import duckdb


def audit(config_path, report_path):
    root = Path(__file__).resolve().parents[1]
    config = json.loads(config_path.read_text(encoding="utf-8-sig"))
    inputs = {key: root / config["inputs"][key] for key in ("interactions", "catalog")}
    if report_path.exists():
        raise FileExistsError(report_path)
    started = time.perf_counter()
    report = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "all rows; no sampling or input modification",
        "duckdb_version": duckdb.__version__,
        "inputs": {key: {"path": str(path.relative_to(root)),
                         "bytes": path.stat().st_size,
                         "mtime_ns": path.stat().st_mtime_ns} for key, path in inputs.items()},
        "date_rule": "Trim whitespace, require YYYY-MM-DD and a real calendar date. Temporal anomalies reported separately.",
    }
    with tempfile.TemporaryDirectory(prefix="steamrec_audit_") as temp, duckdb.connect() as con:
        con.execute("SET memory_limit = ?", [config["execution"]["memory_limit"]])
        con.execute("SET threads = ?", [config["execution"]["threads"]])
        con.execute("SET temp_directory = ?", [temp])
        con.execute("SET preserve_insertion_order = false")
        def rows(sql):
            cursor = con.execute(sql)
            names = [col[0] for col in cursor.description]
            return [dict(zip(names, row)) for row in cursor.fetchall()]

        con.execute("""
            CREATE TEMP TABLE reviews AS
            SELECT TRY_CAST(user_id AS BIGINT) AS user_id,
                   TRY_CAST(app_id AS BIGINT) AS app_id,
                   date AS raw_date,
                   CASE WHEN regexp_full_match(trim(date), '[0-9]{4}-[0-9]{2}-[0-9]{2}')
                        THEN CAST(try_strptime(trim(date), '%Y-%m-%d') AS DATE) END AS review_date,
                   lower(trim(is_recommended)) AS label
            FROM read_csv(?, header=true, all_varchar=true)
        """, [str(inputs["interactions"])])
        report["review_dates"] = rows("""
            SELECT count(*) AS rows,
                   count(*) FILTER (WHERE raw_date IS NULL OR trim(raw_date) = '') AS missing,
                   count(*) FILTER (WHERE review_date IS NULL) AS invalid_total,
                   count(*) FILTER (WHERE raw_date IS NOT NULL AND trim(raw_date) <> ''
                     AND NOT regexp_full_match(trim(raw_date), '[0-9]{4}-[0-9]{2}-[0-9]{2}')) AS invalid_format,
                   count(*) FILTER (WHERE regexp_full_match(trim(raw_date), '[0-9]{4}-[0-9]{2}-[0-9]{2}')
                     AND review_date IS NULL) AS invalid_calendar_date,
                   count(*) FILTER (WHERE raw_date <> trim(raw_date)) AS surrounding_whitespace,
                   min(review_date) AS min_date, max(review_date) AS max_date,
                   count(*) FILTER (WHERE user_id IS NULL OR user_id < 0 OR app_id IS NULL OR app_id <= 0) AS invalid_id_rows
            FROM reviews
        """)[0]
        report["invalid_review_date_examples"] = rows("""
            SELECT raw_date, count(*) AS rows FROM reviews WHERE review_date IS NULL
            GROUP BY raw_date ORDER BY rows DESC, raw_date LIMIT 10
        """)
        print("Full review date scan complete; checking user-game pairs.", flush=True)
        con.execute("""
            CREATE TEMP TABLE repeated_pairs AS
            SELECT user_id, app_id, count(*) AS occurrences,
                   min(review_date) AS first_date, max(review_date) AS last_date,
                   count(DISTINCT label) AS distinct_labels
            FROM reviews
            WHERE user_id IS NOT NULL AND user_id >= 0 AND app_id IS NOT NULL AND app_id > 0
            GROUP BY user_id, app_id HAVING count(*) > 1
        """)
        report["duplicate_pairs"] = rows("""
            SELECT count(*) AS repeated_pair_groups,
                   coalesce(sum(occurrences), 0) AS rows_in_repeated_pairs,
                   coalesce(sum(occurrences - 1), 0) AS excess_rows,
                   coalesce(max(occurrences), 0) AS max_occurrences,
                   count(*) FILTER (WHERE distinct_labels > 1) AS conflicting_label_groups
            FROM repeated_pairs
        """)[0]
        report["duplicate_pair_examples"] = rows("""
            SELECT * FROM repeated_pairs ORDER BY occurrences DESC, user_id, app_id LIMIT 10
        """)
        con.execute("""
            CREATE TEMP TABLE games AS
            SELECT TRY_CAST(app_id AS BIGINT) AS app_id, date_release AS raw_date,
                   CASE WHEN regexp_full_match(trim(date_release), '[0-9]{4}-[0-9]{2}-[0-9]{2}')
                        THEN CAST(try_strptime(trim(date_release), '%Y-%m-%d') AS DATE) END AS release_date
            FROM read_csv(?, header=true, all_varchar=true)
        """, [str(inputs["catalog"])])
        report["release_dates"] = rows("""
            SELECT count(*) AS rows,
                   count(*) FILTER (WHERE raw_date IS NULL OR trim(raw_date) = '') AS missing,
                   count(*) FILTER (WHERE release_date IS NULL) AS invalid_total,
                   count(*) FILTER (WHERE raw_date IS NOT NULL AND trim(raw_date) <> ''
                     AND NOT regexp_full_match(trim(raw_date), '[0-9]{4}-[0-9]{2}-[0-9]{2}')) AS invalid_format,
                   count(*) FILTER (WHERE regexp_full_match(trim(raw_date), '[0-9]{4}-[0-9]{2}-[0-9]{2}')
                     AND release_date IS NULL) AS invalid_calendar_date,
                   min(release_date) AS min_date, max(release_date) AS max_date
            FROM games
        """)[0]
        report["invalid_release_date_examples"] = rows("""
            SELECT app_id, raw_date FROM games WHERE release_date IS NULL ORDER BY app_id LIMIT 10
        """)
        report["temporal_anomalies"] = rows("""
            SELECT count(*) AS review_rows_before_release,
                   count(DISTINCT r.app_id) AS games_with_reviews_before_release
            FROM reviews r JOIN games g USING (app_id) WHERE r.review_date < g.release_date
        """)[0]
        report["before_release_examples"] = rows("""
            SELECT r.app_id, g.release_date, min(r.review_date) AS earliest_review,
                   count(*) AS review_rows
            FROM reviews r JOIN games g USING (app_id) WHERE r.review_date < g.release_date
            GROUP BY r.app_id, g.release_date ORDER BY review_rows DESC, r.app_id LIMIT 10
        """)
        counts = report["review_dates"]
        assert counts["invalid_total"] == counts["missing"] + counts["invalid_format"] + counts["invalid_calendar_date"]
        counts = report["release_dates"]
        assert counts["invalid_total"] == counts["missing"] + counts["invalid_format"] + counts["invalid_calendar_date"]
    for key, path in inputs.items():
        assert path.stat().st_size == report["inputs"][key]["bytes"]
        assert path.stat().st_mtime_ns == report["inputs"][key]["mtime_ns"]
    report["elapsed_seconds"] = round(time.perf_counter() - started, 2)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2, default=str)
    print(json.dumps(report, ensure_ascii=False, indent=2, default=str), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    audit(args.config, args.report)
