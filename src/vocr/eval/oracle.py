"""Phân loại lỗi OCR theo âm tiết và CER Oracle (Bước 2).

Thiết kế: `docs/designs/step2_error_analysis.md`, mục B. Thuần Python, chạy CPU.

- `align_syllables`: căn chỉnh hai dãy âm tiết bằng Levenshtein có trọng số
  (thay thế = khoảng cách ký tự chuẩn hóa) cộng phép tách/gộp từ k:1, 1:k.
- `classify_pair`: gán nhóm a1/a2/b/c/seg cho từng cặp đã căn.
- `analyze_line`: phân loại một dòng + sửa oracle theo vị trí ký tự, có chốt
  chặn "không để oracle làm CER tăng".
- `summarize`: gộp số liệu corpus, bootstrap theo cụm tài liệu.
"""
import random
import re
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Optional

from vocr.eval.metrics import edit_distance, normalize_text
from vocr.nesy.syllables import is_valid_syllable

# Chi phí nguyên (tránh so sánh số thực khi hòa điểm): chèn/xóa = SCALE,
# thay thế = SCALE * khoảng cách ký tự chuẩn hóa, tách/gộp = SEG_BASE + ...
SCALE = 1000
SEG_BASE = 1
MAX_SEG = 3            # gộp/tách tối đa 3 âm tiết
SEG_MAX_DIST = 0.34    # tách/gộp "gần đúng" chỉ khi nối lại sai ≤ 1/3 ký tự

# Ưu tiên khi hòa điểm: thay thế > tách/gộp > xóa > chèn (tất định)
_OP_PRIORITY = {"sub": 0, "seg": 1, "del": 2, "ins": 3}

# Dấu thanh tiếng Việt ở dạng tổ hợp (NFD): huyền, sắc, ngã, hỏi, nặng
_TONE_MARKS = {"̀", "́", "̃", "̉", "̣"}

# Nhóm lỗi dùng cho oracle
ORACLE_MAIN = {"a1", "seg_a1"}                 # FSM chặt sửa được về nguyên tắc
ORACLE_A = ORACLE_MAIN | {"a2", "seg_a2"}      # + lối thoát tên riêng hoàn hảo
ORACLE_A_INS = ORACLE_A | {"c_ins_invalid"}    # + xóa rác chèn thêm


@dataclass(frozen=True)
class Tok:
    text: str
    start: int
    end: int


@dataclass
class Pair:
    op: str                      # match | sub | del | ins | seg
    ref: tuple                   # tuple[Tok, ...] (rỗng nếu ins)
    hyp: tuple                   # tuple[Tok, ...] (rỗng nếu del)
    group: str = ""              # xem classify_pair
    detail: str = ""             # nhãn phụ cho sub: case/punct/tone/diacritic/letter/other

    @property
    def ref_text(self) -> str:
        return " ".join(t.text for t in self.ref)

    @property
    def hyp_text(self) -> str:
        return " ".join(t.text for t in self.hyp)

    def char_edits(self) -> int:
        """Đóng góp (xấp xỉ) vào edit ký tự; chèn/xóa tính cả 1 dấu cách."""
        if self.op == "match":
            return 0
        if self.op == "del":
            return len(self.ref_text) + 1
        if self.op == "ins":
            return len(self.hyp_text) + 1
        return edit_distance(self.ref_text, self.hyp_text)


def tokenize(text: str) -> list[Tok]:
    """Âm tiết = khối không chứa khoảng trắng, giữ vị trí ký tự trong chuỗi."""
    return [Tok(m.group(), m.start(), m.end()) for m in re.finditer(r"\S+", text)]


def is_invalid(tok: str, allow_domain_tokens: bool = True) -> bool:
    # Kiểm U+FFFD tường minh, không phụ thuộc hành vi của is_valid_syllable.
    return "�" in tok or not is_valid_syllable(tok, allow_domain_tokens=allow_domain_tokens)


def _norm_dist(a: str, b: str) -> float:
    if a == b:
        return 0.0
    return edit_distance(a, b) / max(len(a), len(b))


def _sub_cost(a: str, b: str) -> int:
    return 0 if a == b else max(1, round(SCALE * _norm_dist(a, b)))


def _seg_cost(joined_ref: str, joined_hyp: str) -> Optional[int]:
    d = _norm_dist(joined_ref, joined_hyp)
    if d > SEG_MAX_DIST:
        return None
    return SEG_BASE + round(SCALE * d)


def align_syllables(ref: list[Tok], hyp: list[Tok]) -> list[Pair]:
    """Levenshtein có trọng số trên âm tiết, kèm phép tách/gộp k:1 và 1:k."""
    n, m = len(ref), len(hyp)
    INF = float("inf")
    cost = [[INF] * (m + 1) for _ in range(n + 1)]
    back: list[list[Optional[tuple]]] = [[None] * (m + 1) for _ in range(n + 1)]
    cost[0][0] = 0

    def relax(i, j, c, op, di, dj):
        if c < cost[i][j] or (c == cost[i][j] and back[i][j] is not None
                              and _OP_PRIORITY[op] < _OP_PRIORITY[back[i][j][0]]):
            cost[i][j] = c
            back[i][j] = (op, di, dj)

    for i in range(n + 1):
        for j in range(m + 1):
            if i == 0 and j == 0:
                continue
            if i > 0 and j > 0:
                relax(i, j, cost[i - 1][j - 1] + _sub_cost(ref[i - 1].text, hyp[j - 1].text), "sub", 1, 1)
            for k in range(2, MAX_SEG + 1):
                # gộp: k âm tiết nhãn -> 1 âm tiết dự đoán
                if i >= k and j >= 1:
                    c = _seg_cost("".join(t.text for t in ref[i - k:i]), hyp[j - 1].text)
                    if c is not None:
                        relax(i, j, cost[i - k][j - 1] + c, "seg", k, 1)
                # tách: 1 âm tiết nhãn -> k âm tiết dự đoán
                if i >= 1 and j >= k:
                    c = _seg_cost(ref[i - 1].text, "".join(t.text for t in hyp[j - k:j]))
                    if c is not None:
                        relax(i, j, cost[i - 1][j - k] + c, "seg", 1, k)
            if i > 0:
                relax(i, j, cost[i - 1][j] + SCALE, "del", 1, 0)
            if j > 0:
                relax(i, j, cost[i][j - 1] + SCALE, "ins", 0, 1)

    pairs = []
    i, j = n, m
    while i > 0 or j > 0:
        op, di, dj = back[i][j]
        r, h = tuple(ref[i - di:i]), tuple(hyp[j - dj:j])
        if op == "sub" and r[0].text == h[0].text:
            op = "match"
        pairs.append(Pair(op, r, h))
        i, j = i - di, j - dj
    pairs.reverse()
    return pairs


def _strip_marks(s: str, keep_vowel_marks: bool) -> str:
    """Bỏ dấu (NFD). keep_vowel_marks=True: chỉ bỏ 5 dấu thanh, giữ mũ/móc/trăng."""
    out = []
    for ch in unicodedata.normalize("NFD", s):
        if unicodedata.combining(ch):
            if keep_vowel_marks and ch not in _TONE_MARKS:
                out.append(ch)
            continue
        out.append(ch)
    s = unicodedata.normalize("NFC", "".join(out))
    # đ là chữ riêng trong Unicode nhưng về hình là d + gạch -> coi như dấu
    return s if keep_vowel_marks else s.replace("đ", "d").replace("Đ", "D")


def _strip_punct(s: str) -> str:
    return re.sub(r"^[\W_]+|[\W_]+$", "", s)


def sub_detail(r: str, h: str) -> str:
    """Nhãn phụ cho một phép thay thế (dùng cho cả nhóm a và b).

    case: chỉ khác hoa/thường · punct: chỉ khác dấu câu dính kèm ·
    tone: chỉ khác dấu thanh (b1) · diacritic: khác dấu thanh và/hoặc mũ, móc,
    trăng, gạch đ (b1) · letter: khác chữ cái (b2) · other: có U+FFFD.
    """
    if "�" in h:
        return "other"
    if r.lower() == h.lower():
        return "case"
    if _strip_punct(r).lower() == _strip_punct(h).lower():
        return "punct"
    rl, hl = _strip_punct(r).lower(), _strip_punct(h).lower()
    if _strip_marks(rl, True) == _strip_marks(hl, True):
        return "tone"
    if _strip_marks(rl, False) == _strip_marks(hl, False):
        return "diacritic"
    return "letter"


def classify_pair(p: Pair, allow_domain_tokens: bool = True) -> Pair:
    inv = lambda s: is_invalid(s, allow_domain_tokens)
    if p.op == "match":
        p.group = "ok"
    elif p.op == "del":
        p.group = "c_del"
    elif p.op == "ins":
        p.group = "c_ins_invalid" if inv(p.hyp_text) else "c_ins_valid"
    elif p.op == "sub":
        r, h = p.ref[0].text, p.hyp[0].text
        p.detail = sub_detail(r, h)
        if inv(h):
            p.group = "a2" if inv(r) else "a1"
        else:
            p.group = "b"
    elif p.op == "seg":
        if any(inv(t.text) for t in p.hyp):
            p.group = "seg_a2" if any(inv(t.text) for t in p.ref) else "seg_a1"
        else:
            p.group = "seg_valid"
    return p


def b_subgroup(p: Pair) -> str:
    """Nhóm b tách thêm: b1 = sai dấu (thanh/mũ/móc), b2 = sai chữ cái."""
    return {"tone": "b1", "diacritic": "b1", "letter": "b2"}.get(p.detail, "b_" + p.detail)


def apply_oracle(hyp_text: str, pairs: list[Pair], groups: set) -> str:
    """Sửa `hyp_text` tại đúng vị trí ký tự của các cặp thuộc `groups`."""
    edits = []
    for p in pairs:
        if p.group not in groups or not p.hyp:
            continue
        start, end = p.hyp[0].start, p.hyp[-1].end
        new = "" if p.op == "ins" else p.ref_text
        if p.op == "ins":
            # xóa luôn một khoảng trắng liền kề để không để lại dấu cách thừa
            if end < len(hyp_text) and hyp_text[end] == " ":
                end += 1
            elif start > 0 and hyp_text[start - 1] == " ":
                start -= 1
        edits.append((start, end, new))
    out = hyp_text
    for start, end, new in sorted(edits, reverse=True):
        out = out[:start] + new + out[end:]
    return out


@dataclass
class LineResult:
    ref: str
    hyp: str
    pairs: list
    ref_len: int
    edits_raw: int
    edits: dict = field(default_factory=dict)       # mức oracle -> số edit sau chốt chặn
    worse: dict = field(default_factory=dict)       # mức oracle -> sửa xong có tệ hơn không
    corrected: dict = field(default_factory=dict)   # mức oracle -> chuỗi sau sửa
    document: str = ""
    image: str = ""


LEVELS = (("oracle", ORACLE_MAIN), ("oracle_a", ORACLE_A), ("oracle_a_ins", ORACLE_A_INS))


def analyze_line(ref: str, hyp: str, allow_domain_tokens: bool = True,
                 document: str = "", image: str = "") -> LineResult:
    ref, hyp = normalize_text(ref), normalize_text(hyp)
    pairs = [classify_pair(p, allow_domain_tokens) for p in align_syllables(tokenize(ref), tokenize(hyp))]
    raw = edit_distance(ref, hyp)
    res = LineResult(ref, hyp, pairs, len(ref), raw, document=document, image=image)
    prev = raw
    for name, groups in LEVELS:
        fixed = apply_oracle(hyp, pairs, groups)
        e = edit_distance(ref, fixed)
        res.corrected[name] = fixed
        res.worse[name] = e > prev
        # Oracle được quyền không can thiệp -> lấy min; các mức lồng nhau nên
        # so với mức trước (mức sau luôn ≤ mức trước).
        prev = min(prev, e)
        res.edits[name] = prev
    return res


# ── Tổng hợp ────────────────────────────────────────────────────────────────

def _clustered_bootstrap(lines: list[LineResult], level: str, n_boot: int, seed: int):
    by_doc = defaultdict(lambda: [0, 0, 0])   # doc -> [ref_len, raw, oracle]
    for r in lines:
        acc = by_doc[r.document or r.image or str(id(r))]
        acc[0] += r.ref_len
        acc[1] += r.edits_raw
        acc[2] += r.edits[level]
    docs = list(by_doc.values())
    rng = random.Random(seed)
    deltas = []
    for _ in range(n_boot):
        sample = [docs[rng.randrange(len(docs))] for _ in docs]
        tot = sum(s[0] for s in sample)
        if tot:
            deltas.append((sum(s[1] for s in sample) - sum(s[2] for s in sample)) / tot)
    deltas.sort()
    lo = deltas[int(0.025 * (len(deltas) - 1))]
    hi = deltas[int(0.975 * (len(deltas) - 1))]
    return lo, hi, len(docs)


def summarize(lines: list[LineResult], allow_domain_tokens: bool = True,
              n_boot: int = 10000, seed: int = 0, frag_fn=None) -> dict:
    total = sum(r.ref_len for r in lines)
    raw = sum(r.edits_raw for r in lines)
    s = {
        "n_lines": len(lines), "ref_chars": total,
        "n_documents": len({r.document for r in lines if r.document}),
        "allow_domain_tokens": allow_domain_tokens,
        "cer_raw": raw / total if total else 0.0,
        "lines_with_fffd": sum("�" in r.hyp for r in lines),
    }
    for name, _ in LEVELS:
        e = sum(r.edits[name] for r in lines)
        s[f"cer_{name}"] = e / total if total else 0.0
        s[f"delta_{name}"] = s["cer_raw"] - s[f"cer_{name}"]
        s[f"lines_worse_{name}"] = sum(r.worse[name] for r in lines)
        if n_boot and lines:
            lo, hi, k = _clustered_bootstrap(lines, name, n_boot, seed)
            s[f"delta_{name}_ci95"] = [lo, hi]
            s["bootstrap_clusters"] = k

    groups, char_by_group, b_sub, a1_detail = Counter(), Counter(), Counter(), Counter()
    ref_syl = ref_oov = at_risk = 0
    for r in lines:
        for p in r.pairs:
            groups[p.group] += 1
            char_by_group[p.group] += p.char_edits()
            if p.group == "b":
                b_sub[b_subgroup(p)] += 1
            if p.group == "a1":
                a1_detail[p.detail] += 1
            for t in p.ref:
                ref_syl += 1
                if is_invalid(t.text, allow_domain_tokens):
                    ref_oov += 1
                    if p.op == "match":
                        at_risk += 1
    s["syllable_groups"] = dict(sorted(groups.items()))
    s["char_edits_by_group"] = dict(sorted(char_by_group.items()))
    s["char_edits_decomposed_total"] = sum(char_by_group.values())
    s["char_edits_true_total"] = raw
    s["b_subgroups"] = dict(sorted(b_sub.items()))
    s["a1_detail"] = dict(sorted(a1_detail.items()))
    s["ref_syllables"] = ref_syl
    s["ref_oov_rate"] = ref_oov / ref_syl if ref_syl else 0.0
    s["correct_but_invalid"] = at_risk      # FSM chặt sẽ làm hỏng các âm tiết này
    if frag_fn is not None:
        s["fragmentation"] = fragmentation_by_bucket(lines, frag_fn)
    return s


def fragmentation_by_bucket(lines: list[LineResult], n_tokens_fn) -> dict:
    """Tỉ lệ lỗi của âm tiết NHÃN theo số token BPE (1, 2, 3, 4+)."""
    stats = defaultdict(Counter)
    for r in lines:
        for p in r.pairs:
            for t in p.ref:
                if not re.search(r"\w", t.text):
                    continue   # bỏ khối chỉ có dấu câu
                k = n_tokens_fn(_strip_punct(t.text) or t.text)
                b = str(k) if k <= 3 else "4+"
                stats[b]["n"] += 1
                stats[b]["err"] += p.op != "match"
                stats[b]["a"] += p.group in ORACLE_A
    out = {}
    for b in sorted(stats):
        c = stats[b]
        out[b] = {"n": c["n"], "err_rate": c["err"] / c["n"], "a_rate": c["a"] / c["n"]}
    return out
