import csv
import importlib.util
import tempfile
import unittest
from pathlib import Path

import duckdb

spec = importlib.util.spec_from_file_location(
    "prepare_catalog", Path(__file__).resolve().parents[1] / "scripts/prepare_catalog.py"
)
catalog = importlib.util.module_from_spec(spec)
spec.loader.exec_module(catalog)


class CatalogTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.config = {
            "games_csv": "games.csv", "interactions_parquet": "reviews.parquet",
            "output_dir": "output", "manifest": "experiments/run.json",
            "interaction_scope": "synthetic test", "memory_limit": "256MB", "threads": 1,
        }
        with duckdb.connect() as con:
            con.execute("CREATE TABLE reviews AS SELECT * FROM (VALUES (1, 10, true), (2, 20, false), (3, 999, true)) AS r(user_id, app_id, is_recommended)")
            con.execute(f"COPY reviews TO {catalog.sql_path(self.root / 'reviews.parquet')} (FORMAT PARQUET)")

    def write_games(self, rows):
        with (self.root / "games.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["app_id", "title", "date_release", "win", "mac", "linux"])
            writer.writerows(rows)

    def test_clean_and_audit(self):
        self.write_games([
            ["10", " Game A ", "2022-01-01", " TRUE ", "false", "false"],
            ["20", "", "2022-02-30", "unknown", "false", "true"],
            ["30", "Game C", "", "true", "false", "false"],
            ["1.5", "Bad ID", "2022-01-01", "true", "false", "false"],
            ["0", "Zero ID", "2022-01-01", "true", "false", "false"],
        ])
        before = catalog.fingerprint(self.root / "reviews.parquet")
        report = catalog.prepare(self.config, self.root)
        counts = report["counts"]
        self.assertEqual((counts["catalog_clean_rows"], counts["catalog_rejected_rows"]), (3, 2))
        self.assertEqual(counts["missing_release_date"], 2)
        self.assertEqual(counts["orphan_review_rows"], 1)
        self.assertEqual(counts["reviews_with_missing_release_date"], 1)
        self.assertEqual(before, catalog.fingerprint(self.root / "reviews.parquet"))
        with duckdb.connect() as con:
            rows = con.execute("SELECT app_id, title, win, missing_release_date FROM read_parquet(?) ORDER BY app_id", [str(self.root / "output/games_clean.parquet")]).fetchall()
            self.assertEqual(rows[0], (10, "Game A", True, False))
            self.assertEqual(rows[1], (20, None, None, True))
        with self.assertRaises(FileExistsError):
            catalog.prepare(self.config, self.root)

    def test_duplicate_ids_fail_without_artifacts(self):
        self.write_games([
            ["10", "A", "2022-01-01", "true", "false", "false"],
            ["010", "B", "2022-01-02", "true", "false", "false"],
        ])
        with self.assertRaisesRegex(ValueError, "Duplicate catalog IDs"):
            catalog.prepare(self.config, self.root)
        self.assertFalse((self.root / "output").exists())
        self.assertFalse((self.root / "experiments/run.json").exists())


if __name__ == "__main__":
    unittest.main()
