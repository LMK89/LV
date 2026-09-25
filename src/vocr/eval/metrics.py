"""Độ đo OCR: CER, WER, edit similarity.

Mọi chuỗi được chuẩn hóa Unicode NFC trước khi so sánh — cùng một chữ tiếng Việt
có thể mã hóa dựng sẵn (NFC) hoặc tổ hợp (NFD), nếu không chuẩn hóa thì CER bị
thổi phồng dù hình chữ giống hệt.

- `calculate_cer` / `calculate_wer`: theo từng mẫu (trung bình các mẫu = macro).
- `corpus_cer` / `corpus_wer`: tổng edit / tổng độ dài nhãn (micro) — nên báo cáo
  làm số chính vì không bị các dòng ngắn có CER > 100% chi phối.
"""
import unicodedata


def normalize_text(text: str) -> str:
    return unicodedata.normalize("NFC", text or "")


def edit_distance(s1, s2):
    if len(s1) < len(s2):
        return edit_distance(s2, s1)

    if len(s2) == 0:
        return len(s1)

    previous_row = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    return previous_row[-1]


def _rate(ref_units, hyp_units):
    if len(ref_units) == 0:
        return 0.0 if len(hyp_units) == 0 else 1.0
    return float(edit_distance(ref_units, hyp_units)) / len(ref_units)


def calculate_cer(reference, hypothesis):
    return _rate(list(normalize_text(reference)), list(normalize_text(hypothesis)))


def calculate_wer(reference, hypothesis):
    return _rate(normalize_text(reference).split(), normalize_text(hypothesis).split())


def normalized_edit_similarity(s1, s2):
    s1, s2 = normalize_text(s1), normalize_text(s2)
    if not s1 and not s2:
        return 1.0
    return 1.0 - float(edit_distance(s1, s2)) / max(len(s1), len(s2))


def corpus_cer(references, hypotheses):
    """Tổng khoảng cách ký tự / tổng số ký tự nhãn."""
    edits = total = 0
    for ref, hyp in zip(references, hypotheses):
        ref, hyp = normalize_text(ref), normalize_text(hyp)
        edits += edit_distance(list(ref), list(hyp))
        total += len(ref)
    return edits / total if total else 0.0


def corpus_wer(references, hypotheses):
    """Tổng khoảng cách từ / tổng số từ nhãn."""
    edits = total = 0
    for ref, hyp in zip(references, hypotheses):
        r, h = normalize_text(ref).split(), normalize_text(hyp).split()
        edits += edit_distance(r, h)
        total += len(r)
    return edits / total if total else 0.0
