"""Test hàm tính toán NeSy mask và invalid token IDs trên CPU."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import torch
from transformers import AutoTokenizer

from vocr.nesy.loss import (
    compute_eligible_token_ids,
    compute_invalid_token_ids,
    build_invalid_mask,
)

TOKENIZER_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "tokenizers")


def get_tokenizer():
    assert os.path.exists(TOKENIZER_DIR), f"Directory {TOKENIZER_DIR} does not exist"
    return AutoTokenizer.from_pretrained(TOKENIZER_DIR, local_files_only=True)


def test_florence2_vocab_size(tokenizer=None):
    tok = tokenizer or get_tokenizer()
    assert len(tok) == 51290, f"Expected 51290, got {len(tok)}"


def test_eligible_tokens_count(tokenizer=None):
    tok = tokenizer or get_tokenizer()
    eligible = compute_eligible_token_ids(tok)
    # Đúng 49.885 token hợp lệ (loại trừ token đặc biệt và mảnh byte decode ra U+FFFD)
    assert len(eligible) == 49885, f"Expected 49885, got {len(eligible)}"

    # EOS token (</s>, id 2) không được nằm trong eligible
    eos_id = tok.eos_token_id
    if eos_id is not None:
        assert eos_id not in eligible, f"EOS {eos_id} must not be in eligible"


def test_rule_invalid_tokens_count(tokenizer=None):
    tok = tokenizer or get_tokenizer()
    invalid_ids = compute_invalid_token_ids(tok)
    # Đúng 45.775 token bị phạt theo quy tắc NeSy rule
    assert len(invalid_ids) == 45775, f"Expected 45775, got {len(invalid_ids)}"

    # Kiểm tra token U+FFFD không bị phạt
    ufffd_id = tok.convert_tokens_to_ids("\ufffd")
    if ufffd_id != tok.unk_token_id and ufffd_id in range(len(tok)):
        assert ufffd_id not in invalid_ids


def test_build_invalid_mask_rule(tokenizer=None):
    tok = tokenizer or get_tokenizer()
    mask, invalid_ids, rule_ids = build_invalid_mask(tok, mask_mode="rule")
    assert mask.shape == (51290,), f"Expected shape (51290,), got {mask.shape}"
    assert int(mask.sum().item()) == 45775, f"Expected sum 45775, got {int(mask.sum().item())}"
    assert len(invalid_ids) == 45775
    assert invalid_ids == rule_ids


def test_build_invalid_mask_random_fairness(tokenizer=None):
    tok = tokenizer or get_tokenizer()
    eligible = compute_eligible_token_ids(tok)
    rule_mask, rule_ids, _ = build_invalid_mask(tok, mask_mode="rule")

    for seed in [1, 2, 42]:
        mask, ids, rule_forgiven_ids = build_invalid_mask(tok, mask_mode="random", mask_seed=seed)
        assert mask.shape == (51290,)
        assert int(mask.sum().item()) == 45775
        assert len(ids) == 45775

        # 1. Toàn bộ token bị phạt của random PHẢI thuộc eligible (không phạt token đặc biệt / mảnh byte)
        assert ids.issubset(eligible), "Random mask must only sample from eligible tokens"

        # 2. Không bao giờ phạt EOS (</s>)
        eos_id = tok.eos_token_id
        if eos_id is not None:
            assert eos_id not in ids, f"EOS id {eos_id} must never be penalized in random mask"

        # 3. Rule forgiven ids luôn là rule_ids (dùng chung cho cả 2 điều kiện)
        assert rule_forgiven_ids == rule_ids

        # 4. Độ tương đồng Jaccard giữa rule và random trong khoảng 0.83 - 0.87
        jaccard = len(rule_ids & ids) / len(rule_ids | ids)
        assert 0.83 <= jaccard <= 0.87, f"Expected Jaccard ~0.85, got {jaccard:.3f}"


def test_reproducibility_random_seed(tokenizer=None):
    tok = tokenizer or get_tokenizer()
    mask1, ids1, _ = build_invalid_mask(tok, mask_mode="random", mask_seed=42)
    mask2, ids2, _ = build_invalid_mask(tok, mask_mode="random", mask_seed=42)
    mask3, ids3, _ = build_invalid_mask(tok, mask_mode="random", mask_seed=999)

    assert (mask1 == mask2).all()
    assert ids1 == ids2
    assert ids1 != ids3


if __name__ == "__main__":
    print("Testing Florence-2 NeSy mask computation on CPU...")
    tok = get_tokenizer()
    print("- Testing vocab size...")
    test_florence2_vocab_size(tok)
    print("  PASS (vocab_size=51290)")

    print("- Testing eligible tokens count...")
    test_eligible_tokens_count(tok)
    print("  PASS (eligible=49885, no special tokens)")

    print("- Testing rule invalid tokens count...")
    test_rule_invalid_tokens_count(tok)
    print("  PASS (penalised=45775/51290)")

    print("- Testing build_invalid_mask (rule)...")
    test_build_invalid_mask_rule(tok)
    print("  PASS (mask_sum=45775)")

    print("- Testing build_invalid_mask (fair random control across seeds)...")
    test_build_invalid_mask_random_fairness(tok)
    print("  PASS (fair control: subset of eligible, no EOS, Jaccard ~0.85)")

    print("- Testing reproducibility...")
    test_reproducibility_random_seed(tok)
    print("  PASS (reproducible with seed)")

    print("\nALL NESY MASK TESTS PASSED!")
