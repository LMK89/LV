"""Từ điển âm tiết tiếng Việt và các hàm kiểm tra dùng chung.

- `VIET_SYLLABLE_SET`: âm tiết hợp lệ từ `vietnamese_syllables.txt`, mở rộng cả
  hai kiểu đặt dấu thanh (hoà/hòa, thuỷ/thủy).
- `is_valid_syllable`, `could_start_valid_syllable`: kiểm tra cấp chuỗi.
- `context_forgiven_rows`: cửa sổ tha thứ subword cho NeSy Loss.
"""
import os
import re
import unicodedata

# Cặp chuyển đổi dấu thanh giữa kiểu CŨ (hoà, khoá, thuỷ) và kiểu MỚI (hòa, khóa, thủy)
_TONE_PLACEMENT_VARIANTS = (
    # oa
    ("òa", "oà"), ("óa", "oá"), ("ỏa", "oả"), ("õa", "oã"), ("ọa", "oạ"),
    # oe
    ("òe", "oè"), ("óe", "oé"), ("ỏe", "oẻ"), ("õe", "oẽ"), ("ọe", "oẹ"),
    # uy
    ("ùy", "uỳ"), ("úy", "uý"), ("ủy", "uỷ"), ("ũy", "uỹ"), ("ụy", "uỵ"),
    # ua
    ("ùa", "uà"), ("úa", "uá"), ("ủa", "uả"), ("ũa", "uã"), ("ụa", "uạ"),
    # ưa
    ("ừa", "ưà"), ("ứa", "ưá"), ("ửa", "ưả"), ("ữa", "ưã"), ("ựa", "ưạ"),
    # ia
    ("ìa", "ià"), ("ía", "iá"), ("ỉa", "iả"), ("ĩa", "iã"), ("ịa", "iạ"),
)

_TONE_PLACEMENT_MAP = {k: v for k, v in _TONE_PLACEMENT_VARIANTS}
_TONE_PLACEMENT_MAP.update({v: k for k, v in _TONE_PLACEMENT_VARIANTS})


def _expand_tone_variants(syl: str) -> set[str]:
    variants = {syl}
    for k, v in _TONE_PLACEMENT_MAP.items():
        if k in syl:
            variants.add(syl.replace(k, v))
    return variants


_RECEIPT_DOMAIN_TOKENS = frozenset({
    "đ", "đồng", "đ.", "vnd", "vnđ", "usd", "eur",
    "total", "vat", "sum", "subtotal", "cash", "change", "card",
    "combo", "qty", "price", "amount", "date", "time", "no", "id", "tel", "tax",
    "stt", "kh", "hd", "hđ", "kg", "g", "ml", "l", "m", "cm", "mm", "k",
})


def _build_syllable_set() -> frozenset[str]:
    curr_dir = os.path.dirname(os.path.abspath(__file__))
    txt_path = os.path.join(curr_dir, "vietnamese_syllables.txt")
    syllables: set[str] = set()

    raw_items: set[str] = set()
    if os.path.isfile(txt_path):
        with open(txt_path, "r", encoding="utf-8") as f:
            for line in f:
                syl = line.strip().lower()
                if syl:
                    raw_items.add(unicodedata.normalize("NFC", syl))
    else:
        # Thiếu từ điển thì toàn bộ luật sai lệch trong im lặng -> dừng hẳn.
        raise FileNotFoundError(f"Missing syllable dictionary: {txt_path}")

    for item in raw_items:
        syllables.update(_expand_tone_variants(item))

    return frozenset(syllables)


VIET_SYLLABLE_SET: frozenset[str] = _build_syllable_set()


def normalize_syllable(text: str) -> str:
    return unicodedata.normalize("NFC", text.lower().strip())


def is_valid_syllable(syllable: str, allow_domain_tokens: bool = True) -> bool:
    if not syllable:
        return True
    if "\ufffd" in syllable:
        return False
    norm = normalize_syllable(syllable)
    if "\ufffd" in norm:
        return False
    # Ký tự số thuần túy, ký tự đặc biệt, dấu câu
    if re.match(r"^[\d\W_]+$", norm, flags=re.UNICODE):
        return True
    # Bóc tách dấu câu ở đầu/cuối từ (vd: "làm,", "(mê", "linh.", "hỏi?")
    stripped = re.sub(r"^[\W_]+|[\W_]+$", "", norm, flags=re.UNICODE)
    if not stripped:
        return True
    # Âm tiết tiếng Việt chuẩn
    if stripped in VIET_SYLLABLE_SET or norm in VIET_SYLLABLE_SET:
        return True
    # Từ vựng / ký hiệu miền hóa đơn / tài liệu
    if allow_domain_tokens:
        if stripped in _RECEIPT_DOMAIN_TOKENS or norm in _RECEIPT_DOMAIN_TOKENS:
            return True
        # Dạng số lượng / giá tiền kèm đơn vị chuẩn (vd: 100k, 50ml, 2.5kg, 10%, 150.000đ, 10.000.000đ)
        if re.match(r"^\d+([.,]\d+)*(k|đ|g|kg|ml|l|m|cm|mm|%|vnd|vnđ)?$", stripped):
            return True
    return False


def _build_prefix_set(syllables) -> frozenset[str]:
    """Tập mọi tiền tố (theo ký tự) của các âm tiết hợp lệ."""
    prefixes: set[str] = set()
    for syl in syllables:
        for i in range(1, len(syl) + 1):
            prefixes.add(syl[:i])
    return frozenset(prefixes)


# Lấy tiền tố của các âm tiết chữ cái tiếng Việt + từ khóa miền hóa đơn để
# automaton có thể dựng từng token (vd: 'tot' -> 'tota' -> 'total').
VIET_SYLLABLE_PREFIXES: frozenset[str] = _build_prefix_set(
    s for s in (VIET_SYLLABLE_SET | _RECEIPT_DOMAIN_TOKENS) if s.isalpha()
)


def could_start_valid_syllable(text: str, allow_domain_tokens: bool = True) -> bool:
    """True nếu `text` là âm tiết hợp lệ HOẶC là tiền tố của một âm tiết hợp lệ."""
    if not text:
        return True
    if "\ufffd" in text:
        return False
    norm = normalize_syllable(text)
    if "\ufffd" in norm:
        return False
    if re.match(r"^[\d\W_]+$", norm, flags=re.UNICODE):
        return True
    if is_valid_syllable(norm, allow_domain_tokens=allow_domain_tokens):
        return True
    return norm in VIET_SYLLABLE_PREFIXES


def context_forgiven_rows(labels_list, seq_len, invalid_ids, tokenizer, cache,
                          max_window: int = 5):
    """Đánh dấu 'tha thứ' cho subword bị BPE cắt vụn (B1: giảm phạt oan).

    Với mỗi hàng nhãn, một vị trí được tha nếu nó thuộc MỘT dải token liên tiếp
    (độ dài 2..max_window) mà khi decode chung ra chỉ gồm âm tiết hợp lệ. Mở
    rộng cửa sổ từ 2 (chỉ cứu được âm tiết tách 2 token) lên max_window để cứu
    cả âm tiết bị cắt 3-4 mảnh — nguồn chính của sai số xấp xỉ (FPR).

    Trả về list[list[float]] cùng shape [batch, seq_len], 1.0 = được tha.
    `cache` (dict[tuple[int,...], bool]) tái dùng quyết định decode xuyên suốt
    quá trình train vì các dải token giống nhau lặp lại liên tục.
    """
    forgiven_rows = []
    for row in labels_list:
        forgiven = [0.0] * seq_len
        n = min(len(row), seq_len)
        for w in range(2, max_window + 1):
            for i in range(0, n - w + 1):
                window = row[i:i + w]
                # -100 là sentinel bỏ qua; token nằm ngoài dải hợp lệ -> bỏ
                if any(t < 0 for t in window):
                    continue
                # Không token nào invalid riêng lẻ -> không có vấn đề subword
                if all(t not in invalid_ids for t in window):
                    continue
                key = tuple(window)
                forg = cache.get(key)
                if forg is None:
                    try:
                        text = tokenizer.decode(window, skip_special_tokens=True).strip()
                        words = text.split()
                        forg = bool(words) and all(is_valid_syllable(wd) for wd in words)
                    except Exception:
                        forg = False
                    cache[key] = forg
                if forg:
                    for j in range(i, i + w):
                        forgiven[j] = 1.0
        forgiven_rows.append(forgiven)
    return forgiven_rows
