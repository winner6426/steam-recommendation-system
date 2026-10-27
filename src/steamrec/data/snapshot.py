"""Build per-cutoff snapshots (history, ID maps, candidates, targets) from the preprocess DB.

Targets are evaluator-only; sampler and training must never read them.
"""

from pathlib import Path

import duckdb
import numpy as np
import pyarrow.parquet as pq

from steamrec.common import (artifact_hashes, digest, keyed_seed, output_directory,
                             read_json, record, save_array)
from steamrec.data.io import sql_path
from steamrec.data.manifest import fingerprint
from steamrec.data.split import deduplicate_history


def export(con, query, destination):
    destination = Path(destination)
    if destination.exists():
        raise FileExistsError(destination)
    con.execute(f"COPY ({query}) TO {sql_path(destination)} "
                "(FORMAT PARQUET, COMPRESSION ZSTD)")


def csr_from_parquet(source, prefix, n_users, user_column="user_idx"):
    """Input must be ordered by user_idx, item_idx; preserve zero-length users."""
    source, prefix = Path(source), Path(prefix)
    counts = np.zeros(n_users, dtype=np.int64)
    parquet = pq.ParquetFile(source)
    for batch in parquet.iter_batches(columns=[user_column]):
        users = batch.column(0).to_numpy(zero_copy_only=False)
        counts += np.bincount(users, minlength=n_users)
    offsets = np.empty(n_users + 1, dtype=np.int64)
    offsets[0] = 0
    np.cumsum(counts, out=offsets[1:])
    save_array(str(prefix) + "_offsets.npy", offsets)
    size = int(offsets[-1])
    values = np.lib.format.open_memmap(str(prefix) + "_items.npy", mode="w+",
                                      dtype=np.int64, shape=(size,))
    position = 0
    for batch in parquet.iter_batches(columns=["item_idx"]):
        data = batch.column(0).to_numpy(zero_copy_only=False)
        values[position:position + len(data)] = data
        position += len(data)
    values.flush()
    del values


def build_snapshot(config, experiment, stage, cold_app_ids=()):
    if stage not in {"validation", "test"}:
        raise ValueError(stage)
    cutoff = config["T1"] if stage == "validation" else config["T2"]
    end = config["T2"] if stage == "validation" else None
    root = Path(config["paths"]["splits"]) / experiment / stage
    manifest_path = Path(config["paths"]["manifests"]) / f"{experiment}-{stage}.json"
    if root.exists() or manifest_path.exists():
        raise FileExistsError("Snapshot or its manifest already exists")
    pre = read_json(config["paths"]["preprocess_manifest"])
    if config["scope"] == "full" and pre.get("scope") != "full; no LIMIT or sampling":
        raise ValueError("Full run requires a full preprocess manifest")
    database = Path(config["paths"]["database"])
    if not database.is_file():
        raise FileNotFoundError(database)
    declared_db = Path(pre["artifacts"]["database"]).resolve()
    if declared_db != database.resolve():
        raise ValueError("Preprocess manifest describes another database")
    temporary = Path(config["execution"]["temporary_directory"])
    temporary.mkdir(parents=True, exist_ok=True)
    root = output_directory(root)
    with duckdb.connect(str(database), read_only=True) as con:
        con.execute("SET memory_limit = ?", [config["execution"]["memory_limit"]])
        con.execute("SET threads = ?", [config["execution"]["threads"]])
        con.execute("SET temp_directory = ?", [str(temporary)])
        con.execute("SET preserve_insertion_order = false")
        for name in ("interactions", "games", "users", "content"):
            count = con.execute(f"SELECT count(*) FROM {name}").fetchone()[0]
            if count != pre["counts"][name]["stored_rows"]:
                raise ValueError(f"Database row count disagrees with preprocess: {name}")
        history_relation = deduplicate_history(con, cutoff)
        # .query registers a temporary view; works on a read-only database (to_view does not).
        history_relation.query("cutoff_view", "SELECT * FROM cutoff_view")
        con.execute("CREATE TEMP TABLE history AS SELECT * FROM cutoff_view")
        con.execute("CREATE TEMP TABLE hidden(app_id BIGINT)")
        hidden = sorted(set(int(x) for x in cold_app_ids))
        if hidden:
            con.executemany("INSERT INTO hidden VALUES (?)", [(x,) for x in hidden])
        con.execute("""CREATE TEMP VIEW train_history AS SELECT h.* FROM history h
                       WHERE NOT EXISTS (SELECT 1 FROM hidden c WHERE c.app_id=h.app_id)""")
        con.execute("""CREATE TEMP TABLE user_map AS
            SELECT user_id, row_number() OVER (ORDER BY user_id)-1 AS user_idx,
                   count(*) AS review_count,
                   count(*) FILTER (WHERE is_recommended) AS positive_count
            FROM train_history GROUP BY user_id""")
        con.execute("""CREATE TEMP TABLE learned_items AS
            SELECT app_id, row_number() OVER (ORDER BY app_id) AS embedding_idx,
                   count(*) AS train_count,
                   count(*) FILTER (WHERE is_recommended) AS popularity
            FROM train_history GROUP BY app_id""")
        con.execute("""CREATE TEMP TABLE item_map AS
            WITH ids AS (
                SELECT app_id FROM games WHERE date_release <= CAST(? AS DATE)
                UNION SELECT app_id FROM history
            )
            SELECT ids.app_id, row_number() OVER (ORDER BY ids.app_id)-1 AS item_idx,
                   coalesce(l.embedding_idx,0) AS embedding_idx,
                   coalesce(l.train_count,0) AS train_count,
                   coalesce(l.popularity,0) AS popularity,
                   coalesce(g.date_release <= CAST(? AS DATE), false) AS candidate,
                   g.title, g.date_release, coalesce(c.text,'') AS text,
                   coalesce(c.missing_content,true) AS missing_content,
                   EXISTS(SELECT 1 FROM hidden x WHERE x.app_id=ids.app_id) AS cold
            FROM ids LEFT JOIN learned_items l USING(app_id)
            LEFT JOIN games g USING(app_id) LEFT JOIN content c USING(app_id)
        """, [cutoff, cutoff])
        n_users = con.execute("SELECT count(*) FROM user_map").fetchone()[0]
        n_items = con.execute("SELECT count(*) FROM item_map").fetchone()[0]
        if n_users == 0 or n_items == 0:
            raise ValueError("Empty training snapshot")
        export(con, "SELECT * FROM user_map ORDER BY user_idx", root / "users.parquet")
        export(con, "SELECT * FROM item_map ORDER BY item_idx", root / "items.parquet")
        export(con, """SELECT u.user_idx, i.item_idx, h.is_recommended AS label,
                       h.review_id FROM train_history h JOIN user_map u USING(user_id)
                       JOIN item_map i USING(app_id) ORDER BY h.review_id""", root / "train.parquet")
        export(con, """SELECT u.user_idx, i.item_idx FROM history h
                       JOIN user_map u USING(user_id) JOIN item_map i USING(app_id)
                       ORDER BY user_idx, item_idx""", root / "reviewed.parquet")
        export(con, """SELECT u.user_idx, i.item_idx FROM train_history h
                       JOIN user_map u USING(user_id) JOIN item_map i USING(app_id)
                       WHERE NOT h.is_recommended ORDER BY user_idx, item_idx""", root / "explicit.parquet")
        # Future positives are sets; contradictory latest labels are only resolved for history.
        con.execute("""CREATE TEMP TABLE future AS
            SELECT DISTINCT user_id, app_id FROM interactions
            WHERE is_recommended AND date > CAST(? AS DATE)
              AND (? IS NULL OR date <= CAST(? AS DATE))""", [cutoff, end, end])
        con.execute("""CREATE TEMP TABLE target_audit AS
            SELECT f.user_id, f.app_id, u.user_idx, i.item_idx,
                CASE
                  WHEN g.app_id IS NULL THEN 'missing_catalog'
                  WHEN g.date_release IS NULL THEN 'missing_release_date'
                  WHEN g.date_release > CAST(? AS DATE) THEN 'released_after_cutoff'
                  WHEN EXISTS(SELECT 1 FROM history h WHERE h.user_id=f.user_id
                              AND h.app_id=f.app_id) THEN 'already_reviewed'
                  WHEN u.user_idx IS NULL THEN 'unseen_user'
                  WHEN u.positive_count=0 THEN 'no_train_positive'
                  ELSE 'accepted'
                END AS reason
            FROM future f LEFT JOIN games g USING(app_id)
            LEFT JOIN user_map u USING(user_id) LEFT JOIN item_map i USING(app_id)
        """, [cutoff])
        export(con, """SELECT DISTINCT user_idx,item_idx FROM target_audit
                       WHERE reason='accepted' ORDER BY user_idx,item_idx""", root / "targets.parquet")
        reasons = dict(con.execute("SELECT reason,count(*) FROM target_audit GROUP BY reason").fetchall())
        stats = {
            "history_rows": con.execute("SELECT count(*) FROM history").fetchone()[0],
            "train_rows": con.execute("SELECT count(*) FROM train_history").fetchone()[0],
            "train_positive_rows": con.execute("SELECT count(*) FROM train_history WHERE is_recommended").fetchone()[0],
            "train_false_rows": con.execute("SELECT count(*) FROM train_history WHERE NOT is_recommended").fetchone()[0],
            "hidden_rows": con.execute("SELECT count(*) FROM history WHERE app_id IN (SELECT app_id FROM hidden)").fetchone()[0],
            "users": n_users, "items": n_items,
            "learned_items": con.execute("SELECT count(*) FROM learned_items").fetchone()[0],
            "candidate_items": con.execute("SELECT count(*) FROM item_map WHERE candidate").fetchone()[0],
            "eligible_users": con.execute("SELECT count(DISTINCT user_idx) FROM target_audit WHERE reason='accepted'").fetchone()[0],
            "targets_by_reason": reasons,
            "history_users": con.execute("SELECT count(DISTINCT user_id) FROM history").fetchone()[0],
            "future_users": con.execute("SELECT count(DISTINCT user_id) FROM future").fetchone()[0],
        }
    for name in ("reviewed", "explicit", "targets"):
        csr_from_parquet(root / f"{name}.parquet", root / name, n_users)
    snap = {
        "version": 1, "experiment": experiment, "stage": stage, "scope": config["scope"],
        "cutoff": cutoff, "target_end": end, "seed": config["seed"],
        "stats": stats, "cold_app_ids": hidden,
        "protocol": config, "protocol_hash": digest(config),
        "preprocess_identity": fingerprint(config["paths"]["preprocess_manifest"]),
        "database_identity": fingerprint(database),
        "source_inputs": pre["inputs"], "root": str(root),
        "rules": {
            "dedup": "filter cutoff; latest date; same-label ties review_id DESC; conflict stops",
            "future": "distinct positive user-app pairs in evaluation window",
            "candidate": "valid release<=cutoff; exclude all observed reviews incl hidden cold",
            "metadata": "current description/tags snapshot, not historical",
            "mapping": "embedding IDs from train only; item_idx is storage index",
        },
        "artifacts": artifact_hashes(root),
    }
    record(root / "snapshot.json", snap)
    record(manifest_path, snap)
    return root


def cold_items(config):
    with duckdb.connect(config["paths"]["database"], read_only=True) as con:
        rows = con.execute("""SELECT g.app_id FROM games g
            WHERE g.date_release <= CAST(? AS DATE)
              AND EXISTS(SELECT 1 FROM interactions h WHERE h.app_id=g.app_id
                         AND h.date <= CAST(? AS DATE)) ORDER BY g.app_id""",
                           [config["T1"], config["T1"]]).fetchall()
    selected = [int(app) for (app,) in rows if
                keyed_seed(config["seed"], "cold-v1", int(app)) / 2**64 < config["cold"]["fraction"]]
    if not selected:
        raise ValueError("No cold items selected; choose a larger fraction or seed")
    return selected


class Snapshot:
    def __init__(self, directory):
        self.root = Path(directory)
        self.manifest = read_json(self.root / "snapshot.json")
        for name, expected in self.manifest["artifacts"].items():
            if Path(name).name != name:
                raise ValueError("Invalid artifact name")
            actual = fingerprint(self.root / name)
            if (actual["sha256"], actual["bytes"]) != (expected["sha256"], expected["bytes"]):
                raise ValueError(f"Snapshot artifact changed: {name}")
        self.content_manifest = None
        if (self.root / "content.json").is_file():
            self.content_manifest = read_json(self.root / "content.json")
            if self.content_manifest["snapshot_hash"] != digest(self.manifest):
                raise ValueError("Content belongs to a different snapshot")
            for name, expected in self.content_manifest["files"].items():
                actual = fingerprint(self.root / name)
                if actual["sha256"] != expected["sha256"]:
                    raise ValueError(f"Content artifact changed: {name}")
        self.n_users = self.manifest["stats"]["users"]
        self.n_learned_items = self.manifest["stats"]["learned_items"]
        self.items = pq.read_table(self.root / "items.parquet").to_pydict()
        self.n_items = len(self.items["app_id"])
        for name in ("app_id", "item_idx", "embedding_idx", "train_count", "popularity"):
            self.items[name] = np.asarray(self.items[name], dtype=np.int64)
        for name in ("candidate", "missing_content", "cold"):
            self.items[name] = np.asarray(self.items[name], dtype=bool)
        self.candidates = np.flatnonzero(self.items["candidate"])
        self.csr = {}
        for name in ("reviewed", "explicit", "targets"):
            self.csr[name] = (
                np.load(self.root / f"{name}_offsets.npy", mmap_mode="r"),
                np.load(self.root / f"{name}_items.npy", mmap_mode="r"),
            )
        self.content = (np.load(self.root / "content.npy", mmap_mode="r")
                        if (self.root / "content.npy").is_file() else None)

    def row(self, name, user):
        offsets, values = self.csr[name]
        return values[offsets[user]:offsets[user + 1]]

    def train_batches(self, size):
        parquet = pq.ParquetFile(self.root / "train.parquet")
        if parquet.metadata.num_rows != self.manifest["stats"]["train_rows"]:
            raise ValueError("Train Parquet row count disagrees with snapshot")
        for batch in parquet.iter_batches(batch_size=size):
            yield {name: batch.column(j).to_numpy(zero_copy_only=False)
                   for j, name in enumerate(batch.schema.names)}

    def eligible_users(self, size):
        offsets = self.csr["targets"][0]
        for first in range(0, self.n_users, size):
            last = min(first + size, self.n_users)
            users = np.arange(first, last, dtype=np.int64)
            yield users[offsets[first + 1:last + 1] > offsets[first:last]]

    def close(self):
        """Release memory-mapped files (needed on Windows before deleting a snapshot)."""
        self.csr, self.content = {}, None
