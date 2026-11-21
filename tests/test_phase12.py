"""Phase 1-2 on a synthetic two-taste world.

Odd users like "space shooter" games (1-10), even users like "farm puzzle" games (11-20).
Game 10 has no train review (unseen at T1) but has text; game 21 is released after T1.
"""

import gc
import math
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from steam_fixtures import make_config, make_database
from steamrec.common import read_json
from steamrec.data.snapshot import Snapshot, build_snapshot
from steamrec.features.content import build_content
from steamrec.losses.objectives import bpr, weighted_bce
from steamrec.models.hybrid import Hybrid
from steamrec.models.matrix_factorization import MF
from steamrec.training.engine import fit
from steamrec.training.suite import run_test, run_validation, trials_from

SPACE, FARM = list(range(1, 11)), list(range(11, 21))
TEXTS = {**{g: "space shooter sci fi laser" for g in SPACE},
         **{g: "farm puzzle cozy crops" for g in FARM}, 21: "space shooter sequel"}
GAMES = [(g, "2019-01-01") for g in SPACE + FARM] + [(21, "2020-08-01")]


def synthetic_reviews():
    rows, review_id = [], 0

    def add(user, app, date, label):
        nonlocal review_id
        review_id += 1
        rows.append((user, app, date, label, review_id))

    for user in range(1, 41):
        own, other = (SPACE, FARM) if user % 2 else (FARM, SPACE)
        trainable = [g for g in own if g != 10]  # game 10 is never reviewed before T1
        start = user % len(trainable)
        picks = [trainable[(start + j) % len(trainable)] for j in range(6)]
        for j, game in enumerate(picks[:4]):
            add(user, game, f"2020-0{j + 1}-1{user % 9}", True)
        add(user, other[user % len(other)], "2020-05-20", False)
        add(user, picks[4], "2020-08-0{}".format(1 + user % 8), True)   # validation target
        add(user, picks[5], "2020-11-0{}".format(1 + user % 8), True)   # test target
    return rows


class PhaseOneTwoTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        root = cls.tmp.name
        database, pre = make_database(root, synthetic_reviews(), GAMES, TEXTS)
        cls.config, _ = make_config(root, database, pre, epochs=2, batch_size=32,
                                    learning_rate=1.0, embedding_dimension=8)
        cls.val_dir = build_snapshot(cls.config, "fx", "validation")
        cls.test_dir = build_snapshot(cls.config, "fx", "test")
        build_content(cls.val_dir, cls.config)
        build_content(cls.test_dir, cls.config)
        cls.val = Snapshot(cls.val_dir)

    @classmethod
    def tearDownClass(cls):
        cls.val.close()
        del cls.val
        gc.collect()
        cls.tmp.cleanup()

    def item(self, app_id):
        return int(np.flatnonzero(self.val.items["app_id"] == app_id)[0])

    def test_losses_match_formulas(self):
        loss = bpr(torch.tensor([2.0]), torch.tensor([[0.5, 9.0]]), torch.tensor([[1.0, 0.0]]))
        self.assertAlmostEqual(loss.item(), -math.log(1 / (1 + math.exp(-1.5))), places=5)
        self.assertEqual(bpr(torch.tensor([1.0]), torch.tensor([[0.0]]), torch.tensor([[0.0]])).item(), 0)
        bce = weighted_bce(torch.tensor([0.0, 0.0]), torch.tensor([1.0, 0.0]), torch.tensor([1.0, 0.25]))
        self.assertAlmostEqual(bce.item(), math.log(2) * (1 + 0.25) / 2, places=5)

    def test_content_fit_only_on_cutoff_catalog(self):
        meta = self.val.content_manifest
        self.assertEqual(meta["fit_candidate_count"], 20)  # game 21 not released at T1
        content = np.asarray(self.val.content)
        space, farm, space2 = content[self.item(1)], content[self.item(10)], content[self.item(11)]
        self.assertAlmostEqual(float(np.linalg.norm(space)), 1.0, places=5)
        self.assertGreater(space @ farm, space @ space2)  # 1 and 10 share text; 11 differs

    def test_unseen_game_mf_zero_hybrid_content(self):
        torch.manual_seed(0)
        unseen = torch.tensor([self.item(10)])
        self.assertEqual(int(self.val.items["embedding_idx"][self.item(10)]), 0)
        self.assertTrue(torch.all(MF(self.val, 8).item_vectors(unseen) == 0))
        hybrid = Hybrid(self.val, 8).eval()
        self.assertTrue(torch.any(hybrid.item_vectors(unseen) != 0))

    def test_training_learns_taste_groups(self):
        trial = {"name": "mf-bpr-learn", "architecture": "mf", "objective": "bpr"}
        config = {**self.config, "execution": {**self.config["execution"], "epochs": 6}}
        _, reports = fit(self.val, trial, config, Path(self.tmp.name) / "learn", 6)
        first, last = reports[0]["training"], reports[-1]["training"]
        self.assertLess(last["mean_loss"], first["mean_loss"])
        self.assertGreater(last["positive_above_negative_rate"], first["positive_above_negative_rate"])
        self.assertEqual(reports[-1]["evaluation"]["evaluation_coverage"]["evaluated_users"], 40)

    def test_full_suite_validation_then_test(self):
        trials = trials_from(["configs/mf.yaml", "configs/hybrid.yaml"])
        locked = run_validation(self.val_dir, self.config, trials, "suite")
        self.assertEqual(set(locked["selected"]),
                         {"popularity", "mf+bce", "mf+bpr", "hybrid+bce", "hybrid+bpr"})
        self.assertTrue(all(1 <= s["epoch"] <= 2 for n, s in locked["selected"].items() if n != "popularity"))
        comparison = run_test(self.test_dir, self.val_dir, self.config, trials, "suite")
        self.assertEqual(set(comparison["differences"]),
                         {"hybrid+bce minus mf+bce", "hybrid+bpr minus mf+bpr"})
        for row in comparison["rows"]:
            self.assertEqual(row["evaluation"]["evaluation_coverage"]["evaluated_users"], 40)
        runs = Path(self.config["paths"]["reports"]) / "suite"
        self.assertTrue((runs / "validation.csv").is_file() and (runs / "test.csv").is_file())
        with self.assertRaises(FileExistsError):  # test is reported once, never repeated
            run_test(self.test_dir, self.val_dir, self.config, trials, "suite")

    def test_test_refuses_changed_config(self):
        trials = trials_from(["configs/mf.yaml", "configs/hybrid.yaml"])
        run_validation(self.val_dir, self.config, trials, "guard",
                       smoke={"batches": 1, "users": 5})
        changed = {**self.config, "execution": {**self.config["execution"], "learning_rate": 0.5}}
        with self.assertRaises(ValueError):
            run_test(self.test_dir, self.val_dir, changed, trials, "guard")
        self.assertEqual(read_json(Path(self.config["paths"]["reports"]) / "guard" / "locked.json")["scope"],
                         "smoke")

    def test_suite_requires_hybrid_bpr_control(self):
        with self.assertRaises(ValueError):
            trials_from(["configs/mf.yaml"])


if __name__ == "__main__":
    unittest.main()
