"""Phase 2 content vectors: TF-IDF + TruncatedSVD over description + tags, fit per cutoff.

Only candidate games at the snapshot's cutoff are used to fit the encoder (no future catalog).
Every item (including non-candidates) is then transformed; games without text get an
all-zero vector, so the Hybrid content branch contributes nothing for them.
Limitation: description/tags are today's snapshot, not historical text.
"""

import joblib
import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

from steamrec.common import digest, record, save_array
from steamrec.data.manifest import fingerprint


def transform_texts(encoder, texts):
    output = np.zeros((len(texts), encoder["dimensions"]), dtype=np.float32)
    if encoder["vectorizer"] is None:
        return output
    sparse = encoder["vectorizer"].transform(texts)
    dense = sparse.toarray() if encoder["svd"] is None else encoder["svd"].transform(sparse)
    output[:, :dense.shape[1]] = dense
    return normalize(output, copy=False)  # L2 rows; all-zero rows stay zero


def build_content(snapshot_directory, config):
    from steamrec.data.snapshot import Snapshot
    snap = Snapshot(snapshot_directory)
    destinations = [snap.root / name for name in ("content.npy", "encoder.joblib", "content.json")]
    if any(p.exists() for p in destinations):
        raise FileExistsError("Content artifacts already exist")
    texts = snap.items["text"]
    fit_texts = [texts[i] for i in snap.candidates if texts[i].strip()]
    options = config["content"]
    encoder = {"vectorizer": None, "svd": None, "dimensions": options["dimensions"]}
    if fit_texts:
        vectorizer = TfidfVectorizer(
            max_features=options["max_features"], ngram_range=tuple(options["ngram_range"]),
            min_df=options["min_df"], dtype=np.float32, token_pattern=r"(?u)\b\w+\b")
        try:
            fitted = vectorizer.fit_transform(fit_texts)
        except ValueError as error:
            if "empty vocabulary" not in str(error):
                raise
            fitted = None
        if fitted is not None:
            encoder["vectorizer"] = vectorizer
            dimension = min(options["dimensions"], fitted.shape[1] - 1, fitted.shape[0] - 1)
            if dimension >= 1:
                encoder["svd"] = TruncatedSVD(n_components=dimension,
                                              random_state=config["seed"]).fit(fitted)
            elif fitted.shape[1] > options["dimensions"]:
                # Degenerate tiny fixture: still avoid an oversized dense output.
                encoder["svd"] = TruncatedSVD(n_components=1, random_state=config["seed"]).fit(fitted)
    vectors = transform_texts(encoder, texts)
    save_array(destinations[0], vectors)
    with destinations[1].open("xb") as handle:
        joblib.dump(encoder, handle)
    metadata = {
        "version": 1, "cutoff": snap.manifest["cutoff"], "config": options,
        "fit_candidate_count": len(snap.candidates), "fit_nonempty_count": len(fit_texts),
        "zero_vector_items": int((~vectors.any(axis=1)).sum()),
        "shape": list(vectors.shape), "snapshot_hash": digest(snap.manifest),
        "limitation": "description/tags are current snapshot; historical text unavailable",
        "files": {name: fingerprint(snap.root / name) for name in ("content.npy", "encoder.joblib")},
    }
    record(destinations[2], metadata)
    snap.close()
    return metadata
