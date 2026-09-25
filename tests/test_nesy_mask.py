import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import torch
from transformers import AutoTokenizer

from vocr.nesy.loss import compute_invalid_token_ids, build_invalid_mask

TOKENIZER_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "tokenizers")


def get_tokenizer():
    assert os.path.exists(TOKENIZER_DIR), f"Directory {TOKENIZER_DIR} does not exist"
    return AutoTokenizer.from_pretrained(TOKENIZER_DIR, local_files_only=True)


def test_florence2_vocab_size(tok=None):
    tok = tok or get_tokenizer()
    assert len(tok) == 51290, f"Expected 51290, got {len(tok)}"


def test_rule_invalid_tokens_count(tok=None):
    tok = tok or get_tokenizer()
    invalid_ids = compute_invalid_token_ids(tok)
    # Đúng 45.775 token bị phạt theo quy tắc NeSy rule
    # (sau khi đã loại trừ các mảnh byte dở dang decode ra U+FFFD)
    assert len(invalid_ids) == 45775, f"Expected 45775, got {len(invalid_ids)}"

    # Kiểm tra token U+FFFD (id 50284 trong Florence-2) không bị phạt
    ufffd_id = tok.convert_tokens_to_ids("")
    if ufffd_id != tok.unk_token_id and ufffd_id in range(len(tok)):
        assert ufffd_id not in invalid_ids


def test_build_invalid_mask_rule(tok=None):
    tok = tok or get_tokenizer()
    mask, invalid_ids = build_invalid_mask(tok, mask_mode="rule")
    assert mask.shape == (51290,), f"Expected shape (51290,), got {mask.shape}"
    assert int(mask.sum().item()) == 45775, f"Expected sum 45775, got {int(mask.sum().item())}"
    assert len(invalid_ids) == 45775


def test_build_invalid_mask_random(tok=None):
    tok = tok or get_tokenizer()
    mask1, ids1 = build_invalid_mask(tok, mask_mode="random", mask_seed=42)
    mask2, ids2 = build_invalid_mask(tok, mask_mode="random", mask_seed=42)
    mask3, ids3 = build_invalid_mask(tok, mask_mode="random", mask_seed=999)

    # Cùng seed phải hoàn toàn giống nhau
    assert int(mask1.sum().item()) == 45775
    assert ids1 == ids2
    assert (mask1 == mask2).all()

    # Khác seed phải khác nhau nhưng cùng số lượng token
    assert int(mask3.sum().item()) == 45775
    assert ids1 != ids3


if __name__ == "__main__":
    print("Testing Florence-2 NeSy mask computation on CPU...")
    tok = get_tokenizer()
    print("- Testing vocab size...")
    test_florence2_vocab_size(tok)
    print("  PASS (vocab_size=51289)")

    print("- Testing rule invalid tokens count...")
    test_rule_invalid_tokens_count(tok)
    print("  PASS (penalised=45775/51289)")

    print("- Testing build_invalid_mask (rule)...")
    test_build_invalid_mask_rule(tok)
    print("  PASS (mask_sum=45775)")

    print("- Testing build_invalid_mask (random control)...")
    test_build_invalid_mask_random(tok)
    print("  PASS (random mask reproducible and matches size)")

    print("\nALL NESY MASK TESTS PASSED!")
