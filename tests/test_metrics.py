import unicodedata

import pytest

from vocr.eval.metrics import calculate_cer, calculate_wer, corpus_cer, corpus_wer


def test_identical_is_zero():
    assert calculate_cer("tiếng Việt", "tiếng Việt") == 0.0
    assert calculate_wer("tiếng Việt", "tiếng Việt") == 0.0


def test_nfc_nfd_equivalent():
    ref = "người Việt"
    assert calculate_cer(ref, unicodedata.normalize("NFD", ref)) == 0.0


def test_single_substitution():
    assert calculate_cer("học", "hộc") == pytest.approx(1 / 3)
    assert calculate_wer("trường học", "trường hộc") == pytest.approx(1 / 2)


def test_corpus_cer_is_micro_average():
    refs = ["ab", "abcdefghij"]
    hyps = ["xx", "abcdefghij"]
    # macro = (1.0 + 0.0) / 2 = 0.5 ; micro = 2 / 12
    assert corpus_cer(refs, hyps) == pytest.approx(2 / 12)
    assert corpus_wer(["a b", "c d"], ["a x", "c d"]) == pytest.approx(1 / 4)
