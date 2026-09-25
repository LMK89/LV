"""classify_errors_oracle.py — Phân loại lỗi và ước tính trần CER Oracle cho OCR VLM.

Theo thiết kế Bước 2 (docs/designs/step2_error_analysis.md):
- Phân loại lỗi căn chỉnh cấp âm tiết thành 3 nhóm:
    a1: Âm tiết dự đoán không hợp lệ, nhãn hợp lệ (FSM sửa được).
    a2: Âm tiết dự đoán không hợp lệ, nhãn cũng ngoài từ điển (cần cơ chế thoát/bailout).
    b1: Âm tiết hợp lệ nhưng sai dấu thanh (cùng chữ cái gốc).
    b2: Âm tiết hợp lệ nhưng sai chữ cái.
    c:  Lỗi chèn (insertion) hoặc xóa (deletion).
- Sửa lỗi Oracle cho nhóm a1 (thay thế ngược từ nhãn với cơ chế bảo vệ min(edit_before, edit_after)).
- Đo CER thô, CER Oracle và khoảng giảm trần Δ_oracle = CER_raw - CER_oracle.
- Chạy 100% trên CPU, hỗ trợ cờ --dry-run chạy mẫu giả lập độc lập.
"""
import argparse
import json
import os
import re
import sys
import unicodedata

if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass
from collections import defaultdict
from typing import List, Tuple, Dict, Any, Optional

# Đảm bảo nạp được vocr từ thư mục src
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
from vocr.nesy.syllables import is_valid_syllable


# ─── 1. Tiền xử lý & Chuẩn hoá ──────────────────────────────────────────────

def normalize_nfc(text: str) -> str:
    """Chuẩn hóa chuỗi về dạng Unicode NFC."""
    return unicodedata.normalize("NFC", text or "")


def strip_tone_marks(syl: str) -> str:
    """Bóc tách dấu thanh tiếng Việt (ngang, huyền, sắc, hỏi, ngã, nặng) để giữ lại chữ cái gốc."""
    nfd = unicodedata.normalize("NFD", syl.lower())
    # Loại bỏ các combining diacritical marks của dấu thanh:
    # 0x0300 (huyền), 0x0301 (sắc), 0x0303 (ngã), 0x0309 (hỏi), 0x0323 (nặng)
    no_tone = "".join(c for c in nfd if c not in ('\u0300', '\u0301', '\u0303', '\u0309', '\u0323'))
    return unicodedata.normalize("NFC", no_tone)


def is_invalid_token(token: str, allow_domain_tokens: bool = True) -> bool:
    """Kiểm tra một token/âm tiết có vi phạm cấu trúc tiếng Việt không."""
    if not token:
        return False
    if "\ufffd" in token:
        return True
    return not is_valid_syllable(token, allow_domain_tokens=allow_domain_tokens)


def tokenize_with_spans(text: str) -> List[Tuple[str, int, int]]:
    """Tách chuỗi thành các khối non-whitespace kèm vị trí (start, end) ký tự."""
    return [(m.group(0), m.start(), m.end()) for m in re.finditer(r"\S+", text)]


# ─── 2. Khoảng cách chỉnh sửa Levenshtein ─────────────────────────────────────

def char_edit_distance(s1: str, s2: str) -> int:
    """Tính khoảng cách Levenshtein cấp ký tự chuẩn."""
    if s1 == s2:
        return 0
    m, n = len(s1), len(s2)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev = dp[0]
        dp[0] = i
        c1 = s1[i - 1]
        for j in range(1, n + 1):
            temp = dp[j]
            cost = 0 if c1 == s2[j - 1] else 1
            dp[j] = min(dp[j] + 1,      # deletion
                        dp[j - 1] + 1,  # insertion
                        prev + cost)    # substitution
            prev = temp
    return dp[n]


# ─── 3. Căn chỉnh cấp âm tiết có trọng số (Weighted DP) ──────────────────────

def align_syllables_weighted(
    ref_tokens: List[Tuple[str, int, int]],
    hyp_tokens: List[Tuple[str, int, int]]
) -> List[Dict[str, Any]]:
    """
    Căn chỉnh hai dãy âm tiết bằng quy hoạch động có trọng số:
    - Match: 0.0 nếu ref == hyp
    - Substitution: char_edit_distance(ref, hyp) / max(len(ref), len(hyp)) in (0, 1]
    - Insert / Delete: cost = 1.0
    Ưu tiên hòa điểm: Match/Subst > Delete > Insert.
    """
    m, n = len(ref_tokens), len(hyp_tokens)
    dp = [[0.0] * (n + 1) for _ in range(m + 1)]
    backtrack = [[None] * (n + 1) for _ in range(m + 1)]

    for i in range(1, m + 1):
        dp[i][0] = i * 1.0
        backtrack[i][0] = ('del', i - 1, None)
    for j in range(1, n + 1):
        dp[0][j] = j * 1.0
        backtrack[0][j] = ('ins', None, j - 1)

    for i in range(1, m + 1):
        r_tok, _, _ = ref_tokens[i - 1]
        for j in range(1, n + 1):
            h_tok, _, _ = hyp_tokens[j - 1]
            if r_tok == h_tok:
                sub_cost = 0.0
            else:
                sub_cost = char_edit_distance(r_tok, h_tok) / max(len(r_tok), len(h_tok), 1)

            cost_match_sub = dp[i - 1][j - 1] + sub_cost
            cost_del = dp[i - 1][j] + 1.0
            cost_ins = dp[i][j - 1] + 1.0

            # Ưu tiên: match/subst, sau đó del, sau đó ins
            best_cost = cost_match_sub
            best_op = ('sub' if sub_cost > 0 else 'match', i - 1, j - 1)

            if cost_del < best_cost - 1e-9:
                best_cost = cost_del
                best_op = ('del', i - 1, None)
            if cost_ins < best_cost - 1e-9:
                best_cost = cost_ins
                best_op = ('ins', None, j - 1)

            dp[i][j] = best_cost
            backtrack[i][j] = best_op

    # Truy vết các bước căn chỉnh
    ops = []
    i, j = m, n
    while i > 0 or j > 0:
        op_info = backtrack[i][j]
        op, r_idx, h_idx = op_info
        r_entry = ref_tokens[r_idx] if r_idx is not None else None
        h_entry = hyp_tokens[h_idx] if h_idx is not None else None

        ops.append({
            "op": op,
            "ref_tok": r_entry[0] if r_entry else None,
            "ref_span": (r_entry[1], r_entry[2]) if r_entry else None,
            "hyp_tok": h_entry[0] if h_entry else None,
            "hyp_span": (h_entry[1], h_entry[2]) if h_entry else None,
        })

        if op in ('match', 'sub'):
            i -= 1
            j -= 1
        elif op == 'del':
            i -= 1
        elif op == 'ins':
            j -= 1

    ops.reverse()
    return ops


# ─── 4. Phân loại lỗi (Taxonomy) ─────────────────────────────────────────────

def classify_aligned_pair(
    op: str,
    r_tok: Optional[str],
    h_tok: Optional[str],
    allow_domain_tokens: bool = True
) -> str:
    """
    Phân loại từng thao tác căn chỉnh thành các nhóm lỗi:
    - 'match': khớp hoàn hảo
    - 'c_del', 'c_ins': lỗi xóa hoặc chèn
    - 'a1': âm tiết sai không hợp lệ, nhãn hợp lệ (FSM sửa được)
    - 'a2': âm tiết sai không hợp lệ, nhãn cũng ngoài từ điển (cần lối thoát)
    - 'b1': âm tiết hợp lệ nhưng sai dấu thanh (cùng chữ cái gốc)
    - 'b2': âm tiết hợp lệ nhưng sai chữ cái
    """
    if op == 'match':
        return 'match'
    if op == 'del':
        return 'c_del'
    if op == 'ins':
        return 'c_ins'

    # Trường hợp substitution
    h_inv = is_invalid_token(h_tok, allow_domain_tokens=allow_domain_tokens)
    r_inv = is_invalid_token(r_tok, allow_domain_tokens=allow_domain_tokens)

    if h_inv:
        if not r_inv:
            return 'a1'  # FSM can thiệp được
        else:
            return 'a2'  # Tên riêng / ngoài từ điển
    else:
        # Dự đoán là âm tiết hợp lệ
        if strip_tone_marks(h_tok) == strip_tone_marks(r_tok):
            return 'b1'  # Sai dấu thanh
        else:
            return 'b2'  # Sai chữ cái


# ─── 5. Sửa lỗi Oracle (Oracle Correction) ───────────────────────────────────

def apply_oracle_correction(
    hyp_text: str,
    ref_text: str,
    aligned_ops: List[Dict[str, Any]]
) -> Tuple[str, int, int, int]:
    """
    Áp dụng sửa lỗi Oracle cho nhóm a1 (âm tiết sai vi phạm cấu trúc):
    - Thay thế đoạn ký tự hyp[start:end] bằng ref_tok tương ứng.
    - Duyệt ngược từ cuối chuỗi lên đầu để không làm lệch span index.
    - Áp dụng cơ chế an toàn: oracle_edit = min(raw_edit, corrected_edit)
      nhằm tránh bẫy tăng edit distance do lỗi tách/gộp từ.
    Returns:
        (hyp_oracle, raw_edit, corrected_edit, final_oracle_edit)
    """
    raw_edit = char_edit_distance(ref_text, hyp_text)

    # Thu thập các phép thay thế nhóm a1
    substs_to_apply = []
    for item in aligned_ops:
        if item.get("group") == "a1" and item["hyp_span"] and item["ref_tok"]:
            substs_to_apply.append((item["hyp_span"], item["ref_tok"]))

    # Thay thế ngược theo vị trí start giảm dần
    substs_to_apply.sort(key=lambda x: x[0][0], reverse=True)
    corrected_chars = list(hyp_text)
    for (start, end), r_tok in substs_to_apply:
        corrected_chars[start:end] = list(r_tok)

    hyp_oracle = "".join(corrected_chars)
    corrected_edit = char_edit_distance(ref_text, hyp_oracle)
    final_oracle_edit = min(raw_edit, corrected_edit)

    return hyp_oracle, raw_edit, corrected_edit, final_oracle_edit


# ─── 6. Hàm xử lý phân tích một tập dữ liệu ───────────────────────────────────

def analyze_predictions(
    data_pairs: List[Dict[str, Any]],
    allow_domain_tokens: bool = True
) -> Dict[str, Any]:
    """Phân tích danh sách các cặp (reference, predicted)."""
    line_details = []
    total_ref_chars = 0
    total_raw_edit = 0
    total_oracle_edit = 0

    group_counts = defaultdict(int)
    group_examples = defaultdict(list)

    for item in data_pairs:
        ref_raw = normalize_nfc(item.get("reference", ""))
        hyp_raw = normalize_nfc(item.get("predicted", ""))
        img_id = item.get("image", item.get("id", ""))
        doc_id = item.get("document", "")

        ref_tokens = tokenize_with_spans(ref_raw)
        hyp_tokens = tokenize_with_spans(hyp_raw)

        aligned = align_syllables_weighted(ref_tokens, hyp_tokens)

        for op_item in aligned:
            grp = classify_aligned_pair(
                op_item["op"],
                op_item["ref_tok"],
                op_item["hyp_tok"],
                allow_domain_tokens=allow_domain_tokens
            )
            op_item["group"] = grp
            group_counts[grp] += 1
            if grp != "match" and len(group_examples[grp]) < 10:
                group_examples[grp].append({
                    "ref": op_item["ref_tok"],
                    "hyp": op_item["hyp_tok"],
                    "img": img_id
                })

        hyp_oracle, raw_edit, _, final_oracle_edit = apply_oracle_correction(
            hyp_raw, ref_raw, aligned
        )

        ref_len = len(ref_raw)
        total_ref_chars += ref_len
        total_raw_edit += raw_edit
        total_oracle_edit += final_oracle_edit

        line_details.append({
            "image": img_id,
            "document": doc_id,
            "reference": ref_raw,
            "predicted": hyp_raw,
            "predicted_oracle": hyp_oracle,
            "ref_len": ref_len,
            "raw_edit": raw_edit,
            "oracle_edit": final_oracle_edit,
            "aligned_ops": aligned
        })

    cer_raw = total_raw_edit / max(total_ref_chars, 1)
    cer_oracle = total_oracle_edit / max(total_ref_chars, 1)
    delta_oracle = cer_raw - cer_oracle

    total_subst = group_counts["a1"] + group_counts["a2"] + group_counts["b1"] + group_counts["b2"]
    total_errors = total_subst + group_counts["c_del"] + group_counts["c_ins"]

    summary = {
        "total_lines": len(data_pairs),
        "total_ref_chars": total_ref_chars,
        "cer_raw": cer_raw,
        "cer_oracle": cer_oracle,
        "delta_oracle": delta_oracle,
        "delta_oracle_pct_points": delta_oracle * 100.0,
        "total_errors": total_errors,
        "group_counts": dict(group_counts),
        "group_ratios": {k: (v / total_errors if total_errors > 0 else 0.0) for k, v in group_counts.items() if k != "match"},
        "examples": dict(group_examples)
    }

    return {
        "summary": summary,
        "lines": line_details
    }


# ─── 7. Dữ liệu Dry-Run Mẫu ──────────────────────────────────────────────────

MOCK_DRYRUN_DATA = [
    {
        "image": "mock_01.png",
        "reference": "Việt Nam đất nước ta ơi",
        "predicted": "Việt Nam đất nước ta ơi"  # Match hoàn hảo
    },
    {
        "image": "mock_02.png",
        "reference": "chúng ta cùng đi ăn cơm",
        "predicted": "ch\ufffdng ta c\ufffdng đi ăn cơm"  # Nhóm a1: U+FFFD vỡ mã
    },
    {
        "image": "mock_03.png",
        "reference": "hoa hồng nở rộ mùa xuân",
        "predicted": "hoá hồng nở rộ mùa xuân"  # Nhóm b1: sai dấu thanh (hoa -> hoá)
    },
    {
        "image": "mock_04.png",
        "reference": "hàng chục công nhân làm việc",
        "predicted": "hàng chục công nông làm việc"  # Nhóm b2: sai chữ cái hợp lệ
    },
    {
        "image": "mock_05.png",
        "reference": "tổng công ty Cienco 5 tuyên bố",
        "predicted": "tổng công ty Ciencô 5 tuyên bố"  # Nhóm a2: từ mượn/tên riêng
    },
    {
        "image": "mock_06.png",
        "reference": "bác sĩ tiêm một mũi thuốc",
        "predicted": "bác sĩ tiêm mũi thuốc"  # Nhóm c_del: xóa từ "một"
    },
    {
        "image": "mock_07.png",
        "reference": "hôm nay trời nắng đẹp",
        "predicted": "hôm nay thì trời nắng đẹp"  # Nhóm c_ins: chèn từ "thì"
    },
    {
        "image": "mock_08.png",
        "reference": "giá bán là 150.000đ một kg",
        "predicted": "giá bán là 150.000đ một kg"  # Domain tokens hợp lệ
    },
    {
        "image": "mock_09.png",
        "reference": "con ngao sò đang di chuyển",
        "predicted": "con ng\ufffdo s\ufffd đang di chuyển"  # Nhóm a1
    },
    {
        "image": "mock_10.png",
        "reference": "khu phố nhộn nhịp ban đêm",
        "predicted": "khu phố nhộn nhjp ban đêm"  # Nhóm a1: nhjp vi phạm cấu trúc
    }
]


# ─── 8. CLI & Xuất báo cáo ───────────────────────────────────────────────────

def generate_markdown_report(summary: Dict[str, Any]) -> str:
    """Tạo báo cáo Markdown tổng kết kết quả phân loại lỗi."""
    cnt = summary["group_counts"]
    ratios = summary["group_ratios"]
    c_raw = summary["cer_raw"] * 100
    c_orc = summary["cer_oracle"] * 100
    delta = summary["delta_oracle_pct_points"]

    md = f"""# Báo cáo Phân loại Lỗi & Trần CER Oracle (Bước 2)

- **Tổng số câu đánh giá**: {summary['total_lines']}
- **Tổng số ký tự Ground Truth**: {summary['total_ref_chars']}
- **CER ban đầu (Raw)**: {c_raw:.3f}%
- **CER sau khi sửa Oracle (Nhóm a1)**: {c_orc:.3f}%
- **Trần can thiệp tối đa (Δ_oracle)**: **{delta:+.3f} điểm % CER**

---

## 1. Phân bố các nhóm lỗi (Error Taxonomy)

| Nhóm | Ý nghĩa | Số lượng | Tỷ lệ / tổng lỗi | Khả năng can thiệp |
|---|---|---|---|---|
| **a1** | Âm tiết sai không hợp lệ, nhãn hợp lệ | {cnt.get('a1', 0)} | {ratios.get('a1', 0)*100:.1f}% | **FSM / NeSy sửa được** |
| **a2** | Âm tiết sai không hợp lệ, nhãn ngoài từ điển | {cnt.get('a2', 0)} | {ratios.get('a2', 0)*100:.1f}% | Cần cơ chế thoát (Bailout) |
| **b1** | Hợp lệ nhưng sai dấu thanh | {cnt.get('b1', 0)} | {ratios.get('b1', 0)*100:.1f}% | Mô hình ngôn ngữ / LM |
| **b2** | Hợp lệ nhưng sai chữ cái | {cnt.get('b2', 0)} | {ratios.get('b2', 0)*100:.1f}% | Không thể sửa bằng từ điển |
| **c** | Lỗi chèn / xóa ký tự | {cnt.get('c_del', 0) + cnt.get('c_ins', 0)} | {(ratios.get('c_del', 0) + ratios.get('c_ins', 0))*100:.1f}% | Căn chỉnh hình ảnh |

---

## 2. Kết luận định hướng cho Luận văn

- Khoảng trần **Δ_oracle = {delta:.3f}%** cho thấy mức cải thiện tối đa về mặt lý thuyết nếu toàn bộ lỗi vi phạm cấu trúc tiếng Việt được khắc phục hoàn toàn.
- Nếu **Δ_oracle ≥ 2.0%**, đây là bảo chứng khoa học vững chắc để phát triển giải pháp FSM decoding và NeSy Loss ở Bước 3 & Bước 4.
"""
    return md


def main():
    parser = argparse.ArgumentParser(description="Classify OCR Errors & Estimate CER Oracle Ceiling")
    parser.add_argument("--pred_jsonl", type=str, default=None, help="File kết quả dự đoán (.jsonl)")
    parser.add_argument("--split_jsonl", type=str, default="data/splits/val.jsonl", help="File tập chia dữ liệu để ghép document cluster")
    parser.add_argument("--out_dir", type=str, default="outputs/analysis/step2", help="Thư mục xuất báo cáo")
    parser.add_argument("--no_domain_tokens", action="store_true", help="Không cho phép domain tokens trong kiểm tra hợp lệ")
    parser.add_argument("--dry_run", action="store_true", help="Chạy trên 10 cặp mẫu giả lập độc lập trên CPU")
    args = parser.parse_args()

    allow_domain = not args.no_domain_tokens

    if args.dry_run or not args.pred_jsonl:
        print("[DRY-RUN] Chạy phân loại lỗi trên 10 mẫu giả lập độc lập...")
        res = analyze_predictions(MOCK_DRYRUN_DATA, allow_domain_tokens=allow_domain)
    else:
        # Đọc file dự đoán thật
        data = []
        with open(args.pred_jsonl, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    data.append(json.loads(line))
        res = analyze_predictions(data, allow_domain_tokens=allow_domain)

    summary = res["summary"]
    report_md = generate_markdown_report(summary)

    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    with open(os.path.join(args.out_dir, "report.md"), "w", encoding="utf-8") as f:
        f.write(report_md)

    print("\n" + report_md)
    print(f"\n[OK] Đã ghi kết quả phân tích vào: {args.out_dir}")


if __name__ == "__main__":
    main()
