"""Step 2 (Phase 2 only): fit TF-IDF+SVD content vectors inside each snapshot.

Run: python scripts/build_content.py --protocol configs/protocol.yaml \
         --snapshot data/splits/main/validation data/splits/main/test
Each snapshot gets its own encoder fit on games released by its own cutoff.
"""

import argparse
import json

from steamrec.common import protocol
from steamrec.features.content import build_content


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--snapshot", required=True, nargs="+")
    args = parser.parse_args()
    config = protocol(args.protocol)
    for directory in args.snapshot:
        metadata = build_content(directory, config)
        print(directory, json.dumps({k: metadata[k] for k in
                                     ("cutoff", "shape", "fit_nonempty_count", "zero_vector_items")}))


if __name__ == "__main__":
    main()
