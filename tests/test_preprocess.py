import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

import duckdb

from scripts.prepare_data import prepare
from steamrec.data.split import deduplicate_history


class PreprocessTests(unittest.TestCase):
    def fixture(self, root):
        raw = root / "data/raw"
        raw.mkdir(parents=True)
        (raw / "recommendations.csv").write_text(
            "app_id,helpful,funny,date,is_recommended,hours,user_id,review_id\n"
            "10,7,2,2020-02-29,true,0,0,0\n"
            "10,5,1,2020-03-01,false,NaN,0,1\n"
            "20,0,0,2021-02-29,true,1,1,2\n"
            "20,0,0,2020/01/01,true,1,1,3\n"
            "20,0,0,2020-03-01,maybe,1,1,4\n"
            "99,0,0,2020-03-01,true,1,1,5\n", encoding="utf-8")
        (raw / "games.csv").write_text(
            "app_id,title,date_release,win,mac,linux,rating,positive_ratio,user_reviews,price_final,price_original,discount,steam_deck\n"
            "10, Game ,2020-03-01,true,false,false,Very Positive,95,120,9.99,19.99,50,true\n"
            "20,,2021-02-29,true,false,false,Mixed,55,20,0,0,0,false\n", encoding="utf-8")
        (raw / "users.csv").write_text(
            "user_id,products,reviews\n0,2,2\n1,-1,3\n", encoding="utf-8")
        (raw / "games_metadata.json").write_text(
            json.dumps({"app_id": 10, "description": "  text\n here ",
                        "tags": [" Action ", "Action", "", None]}) + "\n" +
            json.dumps({"app_id": 20, "description": None, "tags": None}), encoding="utf-8")
        return {
            "inputs": {"interactions": "data/raw/recommendations.csv", "catalog": "data/raw/games.csv",
                       "users": "data/raw/users.csv", "metadata": "data/raw/games_metadata.json"},
            "outputs": {"database": "data/processed/steamrec.duckdb", "manifest": "experiments/preprocess.json"},
            "execution": {"memory_limit": "128MB", "threads": 1, "temporary_directory": "data/processed/tmp"},
        }

    def test_full_pipeline_persistence_and_quarantine(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = self.fixture(root)
            original = (root / config["inputs"]["interactions"]).read_bytes()
            with contextlib.redirect_stdout(io.StringIO()):
                report = prepare(config, root)
            self.assertEqual(report["counts"]["interactions"],
                             {"input_rows": 6, "stored_rows": 3, "rejected_rows": 3})
            self.assertEqual(report["audit"]["repeated_pairs"], 1)
            self.assertEqual(report["audit"]["orphan_game_reviews"], 1)
            self.assertEqual(report["audit"]["reviews_before_release"], 1)
            self.assertEqual(original, (root / config["inputs"]["interactions"]).read_bytes())
            with duckdb.connect(str(root / config["outputs"]["database"]), read_only=True) as con:
                self.assertEqual({r[0] for r in con.execute("SHOW TABLES").fetchall()},
                                 {"interactions", "games", "users", "content"})
                self.assertEqual(con.execute("SELECT tags FROM content WHERE app_id=10").fetchone()[0], ["Action"])
                self.assertTrue(con.execute("SELECT missing_content FROM content WHERE app_id=20").fetchone()[0])
                self.assertIsNone(con.execute("SELECT hours FROM interactions WHERE review_id=1").fetchone()[0])
                earlier = deduplicate_history(con, "2020-02-29").project("user_id, is_recommended").fetchall()
                later = deduplicate_history(con, "2020-03-01").project("user_id, is_recommended").fetchall()
                self.assertEqual(len(earlier), 1)
                self.assertTrue(earlier[0][1])
                self.assertFalse(next(r for r in later if r[0] == 0)[1])
            expected = {
                "interactions": ["app_id", "helpful", "funny", "date", "is_recommended", "hours", "user_id", "review_id"],
                "games": ["app_id", "title", "date_release", "win", "mac", "linux", "rating", "positive_ratio",
                          "user_reviews", "price_final", "price_original", "discount", "steam_deck"],
                "users": ["user_id", "products", "reviews"],
            }
            with duckdb.connect() as con:
                for name, columns in expected.items():
                    path = report["artifacts"]["parquet"][name]
                    relation = con.read_parquet(path)
                    self.assertEqual(relation.columns, columns)
                    self.assertEqual(relation.count("*").fetchone()[0], report["counts"][name]["stored_rows"])
                self.assertEqual(con.read_parquet(report["artifacts"]["parquet"]["interactions"])
                                 .filter("review_id=0").project("helpful, funny").fetchone(), (7, 2))
                self.assertEqual(con.read_parquet(report["artifacts"]["parquet"]["games"])
                                 .filter("app_id=10").project("rating, positive_ratio, user_reviews, price_final, price_original, discount, steam_deck")
                                 .fetchone(), ("Very Positive", 95, 120, 9.99, 19.99, 50.0, True))
            with self.assertRaises(FileExistsError):
                prepare(config, root)

    def test_duplicate_normalized_catalog_key_fails_before_database(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = self.fixture(root)
            with (root / config["inputs"]["catalog"]).open("a", encoding="utf-8") as handle:
                handle.write("010,Other,2020-01-01,true,false,false,Mixed,50,1,1,1,0,false\n")
            with contextlib.redirect_stdout(io.StringIO()), self.assertRaisesRegex(ValueError, "duplicate normalized key"):
                prepare(config, root)
            self.assertFalse((root / config["outputs"]["database"]).exists())

    def test_same_date_conflict_and_same_label_tie(self):
        with duckdb.connect() as con:
            con.execute("""CREATE TABLE interactions AS SELECT 0 AS user_id, 10 AS app_id,
                DATE '2020-01-01' AS date, true AS is_recommended, 1 AS review_id""")
            con.execute("INSERT INTO interactions VALUES (0,10,'2020-01-01',true,2)")
            self.assertEqual(deduplicate_history(con, "2020-01-01").fetchone()[-1], 2)
            con.execute("INSERT INTO interactions VALUES (0,10,'2020-01-02',false,3)")
            # Future labels do not influence earlier snapshots.
            self.assertTrue(deduplicate_history(con, "2020-01-01").fetchone()[3])
            con.execute("INSERT INTO interactions VALUES (0,10,'2020-01-02',true,4)")
            with self.assertRaisesRegex(ValueError, "Conflicting labels"):
                deduplicate_history(con, "2020-01-02")


    def test_extra_source_column_is_not_silently_discarded(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = self.fixture(root)
            source = root / config["inputs"]["users"]
            source.write_text("user_id,products,reviews,extra\n0,2,2,keep\n", encoding="utf-8")
            with contextlib.redirect_stdout(io.StringIO()), self.assertRaisesRegex(ValueError, "refusing to discard"):
                prepare(config, root)
            self.assertFalse((root / config["outputs"]["database"]).exists())


if __name__ == "__main__":
    unittest.main()
