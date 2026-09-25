"""Unit tests cho module phân loại lỗi và CER Oracle (analysis/classify_errors_oracle.py)."""
import os
import sys

# Đảm bảo nạp được vocr và analysis
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from analysis.classify_errors_oracle import (
    char_edit_distance,
    strip_tone_marks,
    is_invalid_token,
    tokenize_with_spans,
    align_syllables_weighted,
    classify_aligned_pair,
    apply_oracle_correction,
    analyze_predictions,
    MOCK_DRYRUN_DATA,
)


def test_char_edit_distance():
    assert char_edit_distance("hoa", "hoa") == 0
    assert char_edit_distance("hoa", "hoá") == 1
    assert char_edit_distance("Việt", "Viêt") == 1
    assert char_edit_distance("abc", "") == 3


def test_strip_tone_marks():
    assert strip_tone_marks("hoá") == "hoa"
    assert strip_tone_marks("hòa") == "hoa"
    assert strip_tone_marks("hỏa") == "hoa"
    assert strip_tone_marks("người") == "ngươi"
    assert strip_tone_marks("tiến") == "tiên"


def test_is_invalid_token():
    assert is_invalid_token("\ufffd") is True
    assert is_invalid_token("ch\ufffdng") is True
    assert is_invalid_token("xyzq") is True
    assert is_invalid_token("người") is False
    assert is_invalid_token("Việt") is False
    assert is_invalid_token("150.000đ") is False


def test_tokenize_with_spans():
    text = "Việt Nam ơi"
    spans = tokenize_with_spans(text)
    assert len(spans) == 3
    assert spans[0] == ("Việt", 0, 4)
    assert spans[1] == ("Nam", 5, 8)
    assert spans[2] == ("ơi", 9, 11)


def test_align_syllables_weighted():
    ref_tokens = tokenize_with_spans("chúng ta cùng đi")
    hyp_tokens = tokenize_with_spans("chúng ta củng đi")
    aligned = align_syllables_weighted(ref_tokens, hyp_tokens)

    assert len(aligned) == 4
    assert aligned[0]["op"] == "match"
    assert aligned[1]["op"] == "match"
    assert aligned[2]["op"] == "sub"
    assert aligned[2]["ref_tok"] == "cùng"
    assert aligned[2]["hyp_tok"] == "củng"
    assert aligned[3]["op"] == "match"


def test_classify_aligned_pair():
    # Nhóm a1: Dự đoán vỡ mã / sai từ điển, nhãn hợp lệ
    assert classify_aligned_pair("sub", "chúng", "ch\ufffdng") == "a1"
    assert classify_aligned_pair("sub", "người", "ngưqx") == "a1"

    # Nhóm a2: Cả nhãn lẫn dự đoán đều ngoài từ điển (tên riêng)
    assert classify_aligned_pair("sub", "Cienco", "Ciencô") == "a2"

    # Nhóm b1: Cả hai hợp lệ, chỉ khác dấu thanh
    assert classify_aligned_pair("sub", "hoa", "hoá") == "b1"
    assert classify_aligned_pair("sub", "cùng", "củng") == "b1"

    # Nhóm b2: Cả hai hợp lệ nhưng khác chữ cái
    assert classify_aligned_pair("sub", "công", "nông") == "b2"

    # Nhóm c: Chèn / Xóa
    assert classify_aligned_pair("del", "một", None) == "c_del"
    assert classify_aligned_pair("ins", None, "thì") == "c_ins"


def test_oracle_safety_min():
    """Kiểm tra cơ chế min guard: Oracle correction không bao giờ làm tăng edit distance."""
    # Giả sử trường hợp lỗi gộp từ
    ref = "chúng ta ăn cơm"
    hyp = "chúngta ăn cơm"
    tokens_r = tokenize_with_spans(ref)
    tokens_h = tokenize_with_spans(hyp)
    aligned = align_syllables_weighted(tokens_r, tokens_h)
    for a in aligned:
        a["group"] = classify_aligned_pair(a["op"], a["ref_tok"], a["hyp_tok"])

    _, raw_edit, _, final_edit = apply_oracle_correction(hyp, ref, aligned)
    assert final_edit <= raw_edit


def test_analyze_predictions_mock():
    res = analyze_predictions(MOCK_DRYRUN_DATA)
    summary = res["summary"]
    assert summary["total_lines"] == 10
    assert summary["cer_raw"] >= summary["cer_oracle"]
    assert summary["delta_oracle"] >= 0.0
    assert summary["group_counts"]["a1"] > 0
    assert summary["group_counts"]["b1"] > 0
    assert summary["group_counts"]["b2"] > 0


if __name__ == "__main__":
    test_char_edit_distance()
    test_strip_tone_marks()
    test_is_invalid_token()
    test_tokenize_with_spans()
    test_align_syllables_weighted()
    test_classify_aligned_pair()
    test_oracle_safety_min()
    test_analyze_predictions_mock()
    print("ALL ORACLE TESTS PASSED (8/8)!")
