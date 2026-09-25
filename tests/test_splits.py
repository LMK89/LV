import json
import os

import pytest

SPLIT_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "splits")


def _load(name):
    path = os.path.join(SPLIT_DIR, f"{name}.jsonl")
    if not os.path.isfile(path):
        pytest.skip("chưa chạy scripts/split_data.py")
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


@pytest.mark.parametrize("key", ["article", "document", "suffix", "image"])
def test_no_overlap_between_splits(key):
    sets = {s: {r[key] for r in _load(s)} for s in ("train", "val", "test")}
    assert not sets["train"] & sets["val"]
    assert not sets["train"] & sets["test"]
    assert not sets["val"] & sets["test"]


def test_every_label_used_once():
    n = sum(len(_load(s)) for s in ("train", "val", "test"))
    with open(os.path.join(SPLIT_DIR, "..", "raw", "labels.jsonl"), encoding="utf-8") as f:
        assert n == sum(1 for line in f if line.strip())
