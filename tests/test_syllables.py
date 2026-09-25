import unicodedata

from vocr.nesy.syllables import (
    VIET_SYLLABLE_SET, could_start_valid_syllable, is_valid_syllable,
)


def test_dictionary_loaded():
    # từ điển gốc 7.184 dòng, mở rộng thêm biến thể đặt dấu thanh
    assert len(VIET_SYLLABLE_SET) > 7000


def test_common_syllables_valid():
    for w in ["người", "được", "Việt", "nghiêng", "khuya", "hóa", "hoá", "thủy", "thuỷ"]:
        assert is_valid_syllable(w), w


def test_nfd_input_is_normalised():
    assert is_valid_syllable(unicodedata.normalize("NFD", "người"))


def test_punctuation_and_numbers():
    for w in ["làm,", "(Mê", "hỏi?", "150.000đ", "2024", "..."]:
        assert is_valid_syllable(w), w


def test_invalid_strings():
    for w in ["cônk", "nghix", "tion", "xyzq"]:
        assert not is_valid_syllable(w), w


def test_prefixes():
    assert could_start_valid_syllable("ngh")
    assert could_start_valid_syllable("nghiê")
    assert not could_start_valid_syllable("nghx")
