"""Shared config, hashing, seeding and write-once record helpers."""

import hashlib
import json
import platform 
import re
import time
import uuid 
from datetime import date
from pathlib import Path 


import numpy as np
import yaml

from steamrec.data.manifest import fingerprint, write_manifest

def identifier(value): 
    value = str(value)
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", value):
        raise ValueError(f"Unsafe name: {value!r}")

    return value

def read_yaml(path): 
    return yaml.safe_load(Path(path).read_text(encoding = "utf-8-sig"))

def read_json(path):
    return json.loads(Path(path).read_text(encoding = "utf-8"))

def digest(value): 
    raw = json.dumps(value, sort_keys = True, separators = (",", ":"), default = str)
    return hashlib.sha256(raw.encode()).hexdigest()

def keyed_seed(*parts): 
    return int(digest(list(parts))[:16], 16)

def base_protocol(config): 
    return {key: value for key, value in config.items() if key != "run_seed"}

def training_seed(config): 
    return int(config.get("run_seed", config["seed"]))

def protocol(path): 
    config = read_yaml(path)
    if config.get("T1") is None or config.get("T2") is None: 
        raise ValueError("Choose T1/T2 from preprocess monthly report first")
    
    t1, t2 = (date.fromisoformat(str(config[k])) for k in ("T1", "T2"))
    if t1 >= t2: 
        raise ValueError("T1 must be before T2")
    
    config["T1"], config["T2"] = t1.isoformat(), t2.isoformat()
    if config["scope"] not in {"full", "smoke"}:
        raise ValueError("scope must be full or smoke")
    
    seeds = config["repetition"]["seeds"]
    if(len(seeds) < 2 or len(set(seeds)) != len(seeds) or 
       any (isinstance(seed, bool) or not isinstance(seed, int) or seed < 0 for seed in seeds)):
        raise ValueError("Repeated full study needs at least two distinct nonnegative seeds")
   
    s = config["sampler"]
    if s["version"] != "keyed-v1":
        raise ValueError("Unsupported sampler version")
    if min(s["explicit_count"], s["unknown_count"]) < 0:
        raise ValueError("Negative counts must be nonnegative")
    if not 0 < s["unknown_weight"] < s["explicit_weight"]:
        raise ValueError("Unknown weight must be positive and below explicit weight")
    if not 0 <= s["semantic_low"] < s["semantic_high"] <= 1:
        raise ValueError("Invalid semantic quantile band")
    if not 0 <= s["semantic_fraction"] <= 1:
        raise ValueError("Invalid semantic fraction")
    if not 0 < config["cold"]["fraction"] < 1:
        raise ValueError("Cold fraction must be in (0,1)")

    e = config["execution"]
    for key in ("threads", "batch_size", "item_batch_size", "user_batch_size",
                "epochs", "embedding_dimension", "benchmark_batches"):
        if not isinstance(e[key], int) or isinstance(e[key], bool) or e[key] <= 0:
            raise ValueError(f"Invalid {key}")
    if e["checkpoint_every_batches"] < 0 or e["learning_rate"] <= 0 or e["l2"] < 0:
        raise ValueError("Invalid optimizer/checkpoint configuration")
    
    ks = config["evaluation"]["ks"]
    if not ks or any(not isinstance(k, int) or k <= 0 for k in ks):
        raise ValueError("ks must be positive integers")
    if config["evaluation"]["selection_metric"] not in {
        f"{name}@{k}" for name in ("Recall", "NDCG", "HitRate") for k in ks
    }:
        raise ValueError("Selection metric is not computed by evaluator")
    return config

def output_directory(path): 
    path = Path(path)
    path.mkdir(parents = True, exist_ok = False)
    return path

def save_array(path, array):
    with Path(path).open("xb") as handle: 
        np.save(handle, array, allow_pickle=False) 

def runtime(): 
    import duckdb 
    import pyarrow 
    import sklearn 
    import torch 
    import psutil 
    return {
        "python": platform.python_version(), 
        "numpy": np.__version__, 
        "duckdb": duckdb.__version__,
        "pyarrow": pyarrow.__version__,
        "sklearn": sklearn.__version__, 
        "torch": torch.__version__,
        "cuda": torch.version.cuda, 
        "rss_bytes": psutil.Process().memory_info().rss,
        "timestamp_unix": time.time(),
    }

def artifact_hashes(directory):
    return {p.name: fingerprint(p) for p in sorted(Path(directory).iterdir()) if p.is_file()}

def record(path, value): 
    path = Path(path) 
    if path.exists(): 
        raise FileExistsError(path) 
    partial = path.with_name(path.name + ".partial-" + uuid.uuid4().hex)
    write_manifest(partial, value) 

    if path.exists(): 
        raise FileExistsError(path) 
    partial.rename(path)

def record_or_verify(path, value): 
    path = Path(path) 
    if path.is_file():
        if digest(read_json(path)) != digest(value): 
            raise ValueError(f"Existing record has a different identity: {path}")
        return 
    record(path, value)
