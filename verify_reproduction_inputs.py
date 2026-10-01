#!/usr/bin/env python3
"""Verify released instances/checkpoints and the original independent-test lock."""

import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
TRAIN_SEEDS = (20260805, 20260807, 20260808)


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_inputs(root=ROOT):
    for manifest, path_key, expected_count in (("instance_manifest.csv", "package_path", 800),
                                               ("checkpoint_manifest.csv", "package_path", 19)):
        with (root / "protocol" / manifest).open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        if len(rows) != expected_count or len({row[path_key] for row in rows}) != expected_count:
            raise ValueError("Incomplete or duplicate input manifest: " + manifest)
        for row in rows:
            path = root / row[path_key]
            if not path.is_file() or sha256(path) != row["sha256"]:
                raise ValueError("Missing or changed archived input: {0}".format(path))

    protocol = json.loads((root / "protocol/independent_test_v1.json").read_text())
    if protocol["protocol"] != "P0-T_frozen_independent_test_v1":
        raise ValueError("Unexpected independent-test protocol")
    hashes = {}
    for split in ("development", "independent_test"):
        record = protocol["data"][split]
        files = record["files"]
        actual = sorted((root / record["directory"]).glob("*.fjs"))
        if len(files) != 100 or {p.name for p in actual} != {Path(r["path"]).name for r in files}:
            raise ValueError("Independent-test protocol instance list differs: " + split)
        hashes[split] = set()
        for item in files:
            digest = sha256(root / item["path"])
            if digest != item["sha256"]:
                raise ValueError("Instance differs from original lock: " + item["path"])
            hashes[split].add(digest)
    if hashes["development"] & hashes["independent_test"]:
        raise ValueError("Development and independent-test instances overlap")

    # Historical paths in the original lock are mapped to the portable package.
    models = protocol["models"]
    mapping = [("checkpoints/base/reference_save_10_5.pt", models["reference_checkpoint"])]
    for seed in TRAIN_SEEDS:
        mapping.extend([
            ("checkpoints/main/main_seed_{0}.pt".format(seed), models["main"][str(seed)]["final_model"]),
            ("checkpoints/main/main_seed_{0}_conditioned_initial.pt".format(seed), models["main"][str(seed)]["conditioned_initial"]),
            ("checkpoints/static_drl/static_drl_seed_{0}.pt".format(seed), models["static_drl"][str(seed)]),
        ])
    for path, historical in mapping:
        if sha256(root / path) != historical["sha256"]:
            raise ValueError("Checkpoint differs from original test lock: " + path)
    return {"instances": 800, "checkpoints": 19, "development_test_overlap": 0,
            "original_test_models_match": True}


if __name__ == "__main__":
    print(json.dumps(verify_inputs(), indent=2))
    print("PASS frozen input verification")
