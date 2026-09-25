"""Test phân loại lỗi và CER Oracle (docs/designs/step2_error_analysis.md, mục B6)."""
import json
import os
import random
import unicodedata

try:
    import pytest
except ImportError:
    class _MockPytest:
        @staticmethod
        def approx(expected, rel=1e-5, abs_tol=1e-9):
            class _Approx:
                def __init__(self, exp):
                    self.exp = exp
                def __eq__(self, other):
                    diff = abs(other - self.exp)
                    tol = max(rel * abs(self.exp), abs_tol)
                    return diff <= tol
            return _Approx(expected)

        class mark:
            @staticmethod
            def parametrize(argnames, argvalues):
                def decorator(fn):
                    def wrapper(*args, **kwargs):
                        for val in argvalues:
                            if isinstance(val, (tuple, list)) and not isinstance(val, str):
                                fn(*val)
                            else:
                                fn(val)
                    return wrapper
                return decorator
    pytest = _MockPytest()

from vocr.eval.metrics import corpus_cer, edit_distance
from vocr.eval.oracle import analyze_line, b_subgroup, is_invalid, summarize

ROOT = os.path.join(os.path.dirname(__file__), "..")


def groups(line):
    return [p.group for p in line.pairs if p.group != "ok"]


def only_error(line):
    errs = [p for p in line.pairs if p.group != "ok"]
    assert len(errs) == 1, [(p.op, p.ref_text, p.hyp_text, p.group) for p in errs]
    return errs[0]


# 1
def test_identical_line_has_no_error():
    r = analyze_line("Việt Nam là một đất nước", "Việt Nam là một đất nước")
    assert groups(r) == [] and r.edits_raw == 0
    assert all(v == 0 for v in r.edits.values())


# 2
@pytest.mark.parametrize("hyp", ["hóa", "hoá"])
def test_valid_wrong_tone_is_b1(hyp):
    p = only_error(analyze_line("hoa nở", f"{hyp} nở"))
    assert p.group == "b" and b_subgroup(p) == "b1"


def test_wrong_letter_is_b2():
    p = only_error(analyze_line("cây bút", "cây bát"))
    assert p.group == "b" and b_subgroup(p) == "b2"


def test_d_stroke_counts_as_diacritic():
    p = only_error(analyze_line("đi học", "di học"))
    assert p.group == "b" and b_subgroup(p) == "b1"


def test_case_and_punct_subgroups():
    r = analyze_line("Năm nay tuổi,", "năm nay tuổi")
    assert sorted(b_subgroup(p) for p in r.pairs if p.group == "b") == ["b_case", "b_punct"]


# 3
def test_invalid_syllable_is_a1_and_oracle_fixes_it():
    r = analyze_line("người dân", "ngươì dân")
    assert only_error(r).group == "a1"
    assert r.edits_raw > 0 and r.edits["oracle"] == 0
    assert r.corrected["oracle"] == "người dân"


# 4
def test_oov_reference_is_a2_not_in_main_oracle():
    r = analyze_line("ông Obama đến", "ông Obamn đến")
    assert only_error(r).group == "a2"
    assert r.edits["oracle"] == r.edits_raw == 1
    assert r.edits["oracle_a"] == 0


# 5
@pytest.mark.parametrize("tok", ["�", "��", "ú�", "(�", "chng�"])
def test_fffd_always_invalid(tok):
    assert is_invalid(tok)


def test_fffd_substitution_is_a1():
    r = analyze_line("chúng lừa", "chng� l�a")
    assert groups(r) == ["a1", "a1"] and r.edits["oracle"] == 0


# 6
def test_merge_is_segmentation_and_oracle_reaches_zero():
    ref, hyp = "chúng ta ăn", "chúngta ăn"
    r = analyze_line(ref, hyp)
    p = only_error(r)
    assert p.op == "seg" and p.group == "seg_a1"
    assert r.edits_raw == 1 and r.edits["oracle"] == 0
    # oracle ngây thơ (thay chúngta -> chúng, để lộ phép xóa "ta") làm CER TĂNG
    assert edit_distance(ref, "chúng ăn") == 3 > r.edits_raw


def test_split_is_segmentation():
    r = analyze_line("người dân", "ng ười dân")
    p = only_error(r)
    assert p.op == "seg" and p.group == "seg_a1" and r.edits["oracle"] == 0


# 7
def test_deletion_and_insertions():
    r = analyze_line("tôi đi học về", "tôi đi về")
    assert only_error(r).group == "c_del"
    r = analyze_line("trời đẹp", "trời đẹp xyzq")
    assert only_error(r).group == "c_ins_invalid"
    assert r.edits["oracle_a"] == r.edits_raw and r.edits["oracle_a_ins"] == 0
    assert r.corrected["oracle_a_ins"] == "trời đẹp"
    r = analyze_line("trời đẹp", "trời rất đẹp")
    assert only_error(r).group == "c_ins_valid"


# 8
def _corrupt(text, rng):
    """Nhiễu ngẫu nhiên kiểu OCR: đổi dấu, xóa/chèn ký tự, gộp/tách từ, U+FFFD."""
    chars = list(text)
    for _ in range(rng.randint(1, 6)):
        if not chars:
            break
        i = rng.randrange(len(chars))
        op = rng.random()
        if op < 0.3:
            base = unicodedata.normalize("NFD", chars[i])[0]
            chars[i] = unicodedata.normalize("NFC", base + rng.choice(["́", "̀", "̣", ""]))
        elif op < 0.45:
            del chars[i]
        elif op < 0.6:
            chars.insert(i, rng.choice("aăâeêoôơuưy �"))
        elif op < 0.75 and chars[i] == " ":
            del chars[i]
        else:
            chars[i] = rng.choice(["�", "x", "q", " "])
    return "".join(chars)


def _val_refs(n):
    with open(os.path.join(ROOT, "data", "splits", "val.jsonl"), encoding="utf-8") as f:
        return [json.loads(l)["suffix"] for l in f if l.strip()][:n]


def test_oracle_never_increases_cer_and_levels_are_nested():
    rng = random.Random(0)
    lines = []
    for ref in _val_refs(120):
        hyp = _corrupt(ref, rng)
        r = analyze_line(ref, hyp)
        assert r.edits["oracle_a_ins"] <= r.edits["oracle_a"] <= r.edits["oracle"] <= r.edits_raw
        lines.append(r)
    s = summarize(lines, n_boot=0)
    assert s["cer_raw"] == pytest.approx(corpus_cer([l.ref for l in lines], [l.hyp for l in lines]))
    assert 0 <= s["delta_oracle"] <= s["delta_oracle_a"] <= s["delta_oracle_a_ins"]
    # phân rã theo nhóm là cận trên của edit thật
    assert s["char_edits_decomposed_total"] >= s["char_edits_true_total"]


# 9
def test_correction_keeps_untouched_whitespace():
    r = analyze_line("người  dân  ở đây", "ngươì  dân  ở đây")
    assert r.corrected["oracle"] == "người  dân  ở đây"


# 10
def test_nfc_nfd_match():
    ref = "người Việt"
    r = analyze_line(ref, unicodedata.normalize("NFD", ref))
    assert groups(r) == [] and r.edits_raw == 0


# 11
def _docs_lines(n_docs=5, per_doc=3):
    out = []
    for d in range(n_docs):
        for _ in range(per_doc):
            out.append(analyze_line("người dân", "ngươì dân", document=f"doc{d}"))
    return out


def test_bootstrap_degenerate_when_documents_identical():
    s = summarize(_docs_lines(), n_boot=500, seed=1)
    lo, hi = s["delta_oracle_ci95"]
    assert lo == pytest.approx(s["delta_oracle"]) and hi == pytest.approx(s["delta_oracle"])
    assert s["bootstrap_clusters"] == 5


def test_bootstrap_is_reproducible():
    lines = _docs_lines()
    lines += [analyze_line("hoa nở", "hóa nở", document="doc9")]
    a = summarize(lines, n_boot=300, seed=7)["delta_oracle_ci95"]
    b = summarize(lines, n_boot=300, seed=7)["delta_oracle_ci95"]
    assert a == b


def test_domain_token_flag():
    # "vnd" hợp lệ khi bật allow_domain_tokens -> sai thành "vnđ" là b; tắt -> a1 không áp dụng
    # vì cả hai phía đều ngoài từ điển -> a2
    on = only_error(analyze_line("giá 5 usd", "giá 5 vnd"))
    off = only_error(analyze_line("giá 5 usd", "giá 5 vnd", allow_domain_tokens=False))
    assert on.group == "b" and off.group == "a2"


def test_fragmentation_buckets():
    lines = [analyze_line("người dân", "ngươì dân")]
    frag = summarize(lines, n_boot=0, frag_fn=lambda s: len(s.encode("utf-8")))["fragmentation"]
    assert frag["4+"]["n"] == 2 and frag["4+"]["a_rate"] == 0.5
