"""Snapshot split, target audit, sampler and metrics on a 13-review fixture.

Each review row is chosen so every target-exclusion reason appears exactly once.
"""

import gc
import tempfile
import unittest

import numpy as np

from steam_fixtures import make_config, make_database
from steamrec.data.sampling import NegativeSampler
from steamrec.data.snapshot import Snapshot, build_snapshot
from steamrec.evaluation.ranking import evaluate, metrics, row_topk, topk

# (user_id, app_id, date, is_recommended, review_id); T1=2020-06-30, T2=2020-09-30.
REVIEWS = [
    (1, 10, "2020-01-01", True, 1),
    (1, 20, "2020-02-01", False, 2),
    (1, 30, "2020-07-15", True, 3),   # validation target, accepted
    (1, 20, "2020-08-01", True, 4),   # already reviewed before T1
    (1, 40, "2020-08-10", True, 5),   # game released after T1
    (2, 10, "2020-03-01", True, 6),
    (2, 10, "2020-05-01", True, 7),   # repeated pair, dedup keeps latest
    (2, 50, "2020-10-15", True, 8),   # test target
    (3, 20, "2020-04-01", False, 9),  # False-only history
    (3, 30, "2020-07-20", True, 10),  # no train positive at T1
    (4, 50, "2020-07-01", True, 11),  # unseen user at T1
    (4, 30, "2020-11-01", True, 12),  # test target
    (1, 50, "2020-12-01", True, 13),  # test target
]
GAMES = [(10, "2019-01-01"), (20, "2019-01-01"), (30, "2019-01-01"),
         (40, "2020-08-01"), (50, "2019-01-01")]


class SnapshotEvaluationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        database, pre = make_database(cls.tmp.name, REVIEWS, GAMES)
        cls.config, _ = make_config(cls.tmp.name, database, pre)
        cls.val = Snapshot(build_snapshot(cls.config, "fx", "validation"))
        cls.test = Snapshot(build_snapshot(cls.config, "fx", "test"))

    @classmethod
    def tearDownClass(cls):
        del cls.val, cls.test  # release memory-mapped CSR files before Windows can delete them
        gc.collect()
        cls.tmp.cleanup()

    def app_ids(self, snap, items):
        return sorted(int(snap.items["app_id"][i]) for i in items)

    def user_targets(self, snap):
        return {u: self.app_ids(snap, snap.row("targets", u)) for u in range(snap.n_users)
                if len(snap.row("targets", u))}

    def test_validation_history_never_uses_future(self):
        stats = self.val.manifest["stats"]
        self.assertEqual(stats["history_rows"], 4)  # u1:10,20  u2:10 (deduped)  u3:20
        self.assertEqual(stats["users"], 3)
        self.assertEqual(stats["train_false_rows"], 2)

    def test_validation_target_reasons(self):
        reasons = self.val.manifest["stats"]["targets_by_reason"]
        self.assertEqual(reasons, {"accepted": 1, "already_reviewed": 1, "released_after_cutoff": 1,
                                   "no_train_positive": 1, "unseen_user": 1})
        self.assertEqual(list(self.user_targets(self.val).values()), [[30]])
        self.assertEqual(self.val.manifest["stats"]["eligible_users"], 1)

    def test_candidates_respect_release_date(self):
        self.assertEqual(self.app_ids(self.val, self.val.candidates), [10, 20, 30, 50])
        self.assertEqual(self.app_ids(self.test, self.test.candidates), [10, 20, 30, 40, 50])

    def test_unseen_items_get_zero_embedding(self):
        items = self.val.items
        unseen = {int(a) for a, e in zip(items["app_id"], items["embedding_idx"]) if e == 0}
        self.assertEqual(unseen, {30, 50})

    def test_test_snapshot_retrains_to_t2(self):
        self.assertEqual(self.test.n_users, 4)
        self.assertEqual(sorted(self.user_targets(self.test).values()), [[30], [50], [50]])

    def test_sampler_is_reproducible_and_excludes_reviewed(self):
        sampler = NegativeSampler(self.val, self.config["sampler"], seed=42)
        user = 0  # user_id 1; reviewed 10 (True) and 20 (False)
        positive = int(np.flatnonzero(self.val.items["app_id"] == 10)[0])
        ids, weights = sampler.sample(user, positive, epoch=0)
        again, _ = sampler.sample(user, positive, epoch=0)
        np.testing.assert_array_equal(ids, again)
        live = ids[weights > 0]
        explicit = live[weights[weights > 0] == self.config["sampler"]["explicit_weight"]]
        unknown = live[weights[weights > 0] == self.config["sampler"]["unknown_weight"]]
        self.assertEqual(self.app_ids(self.val, explicit), [20])
        self.assertEqual(self.app_ids(self.val, unknown), [30, 50])  # never reviewed, catalog only

    def test_topk_and_metrics(self):
        _, items = topk([1.0, 3.0, 3.0, 2.0], [7, 9, 4, 1], 3)
        self.assertEqual(items.tolist(), [4, 9, 1])  # tie broken by item index
        # row_topk: same rule on positions; -inf (masked) never selected.
        self.assertEqual(row_topk(np.array([2.0, 5.0, 5.0, -np.inf, 1.0]), 3).tolist(), [1, 2, 0])
        self.assertEqual(row_topk(np.array([-np.inf, 1.0]), 3).tolist(), [1])
        m = metrics(np.array([5, 2, 8]), [2, 9], 3)
        self.assertAlmostEqual(m["Recall@3"], 0.5)
        self.assertAlmostEqual(m["NDCG@3"], (1 / np.log2(3)) / (1 + 1 / np.log2(3)))
        self.assertEqual(m["HitRate@3"], 1.0)

    def test_popularity_evaluates_every_eligible_user(self):
        report = evaluate(self.val, None, self.config)
        self.assertEqual(report["evaluation_coverage"]["evaluated_users"], 1)
        # u1 reviewed 10 and 20; the rest tie at popularity 0 and 30 precedes 50.
        self.assertEqual(report["examples"][0]["app_ids"], [30, 50])
        self.assertAlmostEqual(report["groups"]["all"]["Recall@10"], 1.0)
        self.assertEqual(evaluate(self.test, None, self.config)["evaluation_coverage"]["evaluated_users"], 3)


if __name__ == "__main__":
    unittest.main()
