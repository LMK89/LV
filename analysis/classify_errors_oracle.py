"""Bước 2: phân loại lỗi theo âm tiết và tính trần CER Oracle (thuần CPU).

Logic nằm ở `src/vocr/eval/oracle.py`; file này chỉ đọc/ghi và in báo cáo.
Thiết kế: `docs/designs/step2_error_analysis.md`.

Cách chạy:

  # Mẫu mock viết sẵn, không đọc file dự đoán nào
  python analysis/classify_errors_oracle.py --dry-run

  # Trên output của scripts/evaluate.py (khóa reference/predicted/image)
  python analysis/classify_errors_oracle.py \
      --pred_jsonl outputs/results/eval_lam0_s1_<ts>.jsonl \
      --split_jsonl data/splits/val.jsonl \
      --out_dir outputs/analysis/step2/lam0_s1

Tokenizer Florence-2 đọc offline từ `data/tokenizers/tokenizer.json` (cần gói
`tokenizers`, không cần torch/transformers). Thiếu thì bỏ qua phần phân mảnh.
"""
import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))

from vocr.eval.oracle import LEVELS, analyze_line, b_subgroup, summarize  # noqa: E402

# (document, reference, predicted) — mỗi dòng minh họa một nhóm lỗi
MOCK = [
    ("d1", "Việt Nam là một đất nước", "Việt Nam là một đất nước"),        # không lỗi
    ("d1", "người dân rất hiếu khách", "ngươì dân rất hiếu khách"),        # a1 (sai vị trí dấu)
    ("d1", "hoa nở vào mùa xuân", "hóa nở vào mùa xuân"),                  # b1 (sai dấu thanh)
    ("d2", "chúng ta ăn cơm", "chúngta ăn cơm"),                           # seg_a1 (gộp từ)
    ("d2", "ông Obama đến thăm", "ông Obamn đến thăm"),                    # a2 (tên riêng)
    ("d2", "Có khi chúng lừa lúc chủ gà", "Có khi chng� l�a lúc chủ gà"),  # a1 + U+FFFD
    ("d3", "trời hôm nay đẹp", "trời hôm nay đẹp xyzq"),                   # c_ins_invalid
    ("d3", "tôi đi học về muộn", "tôi đi về muộn"),                        # c_del
    ("d3", "cây bút màu xanh", "cây bát màu xanh"),                        # b2 (sai chữ cái)
    ("d3", "Năm nay đã gần 80 tuổi,", "năm nay đã gần 80 tuổi"),           # b_case + b_punct
]


def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def load_frag_fn(tokenizer_dir):
    path = os.path.join(tokenizer_dir, "tokenizer.json") if tokenizer_dir else ""
    if not path or not os.path.isfile(path):
        return None, "không có tokenizer.json"
    try:
        from tokenizers import Tokenizer
    except ImportError:
        return None, "thiếu gói `tokenizers`"
    tk = Tokenizer.from_file(path)
    cache = {}

    def n_tokens(syl):
        if syl not in cache:
            # dấu cách đứng trước: mô phỏng âm tiết ở giữa câu
            cache[syl] = len(tk.encode(" " + syl, add_special_tokens=False).ids)
        return cache[syl]
    return n_tokens, path


def pct(x):
    return f"{100 * x:.2f}"


def render_report(s, lines, n_examples, source):
    out = [f"# Phân loại lỗi & CER Oracle — {source}", ""]
    out += [f"- Số dòng: {s['n_lines']} · ký tự nhãn: {s['ref_chars']} · tài liệu: {s['n_documents']}",
            f"- `allow_domain_tokens`: {s['allow_domain_tokens']}",
            f"- Dòng dự đoán có U+FFFD: {s['lines_with_fffd']}", ""]
    out += ["## CER Oracle (corpus, điểm %)", "",
            "| Mức | Sửa nhóm | CER | Δ | KTC 95% của Δ (bootstrap theo tài liệu) | Dòng mà sửa làm tệ hơn |",
            "|---|---|---|---|---|---|",
            f"| raw | — | {pct(s['cer_raw'])} | — | — | — |"]
    what = {"oracle": "a1 + seg_a1 (**chính**)", "oracle_a": "+ a2, seg_a2", "oracle_a_ins": "+ xóa c_ins_invalid"}
    for name, _ in LEVELS:
        ci = s.get(f"delta_{name}_ci95")
        ci_s = f"[{pct(ci[0])}, {pct(ci[1])}]" if ci else "—"
        out.append(f"| {name} | {what[name]} | {pct(s[f'cer_{name}'])} | {pct(s[f'delta_{name}'])} | {ci_s} | {s[f'lines_worse_{name}']} |")
    out += ["", f"Số cụm bootstrap: {s.get('bootstrap_clusters', '—')}.", ""]

    out += ["## Số âm tiết và edit ký tự theo nhóm", "",
            "| Nhóm | Âm tiết | Edit ký tự (xấp xỉ) |", "|---|---|---|"]
    for g, n in s["syllable_groups"].items():
        out.append(f"| {g} | {n} | {s['char_edits_by_group'].get(g, 0)} |")
    out += ["", f"Tổng edit phân rã {s['char_edits_decomposed_total']} so với edit thật "
            f"{s['char_edits_true_total']} (phân rã là cận trên, chỉ để tham khảo).", ""]
    fmt = lambda d: ", ".join(f"{k}={v}" for k, v in d.items()) or "—"
    out += [f"Nhóm b tách nhỏ (b1 = sai dấu, b2 = sai chữ cái): {fmt(s['b_subgroups'])}", "",
            f"Nhóm a1 theo kiểu sai: {fmt(s['a1_detail'])}", ""]
    out += ["## Rủi ro ngược của FSM chặt", "",
            f"- Âm tiết nhãn ngoài từ điển: {pct(s['ref_oov_rate'])} % ({s['ref_syllables']} âm tiết)",
            f"- Âm tiết đã đúng nhưng ngoài từ điển (FSM chặt sẽ làm hỏng): {s['correct_but_invalid']}", ""]
    if "fragmentation" in s:
        out += ["## Lỗi theo số token BPE của âm tiết nhãn (Florence-2)", "",
                "| Token/âm tiết | Số âm tiết | Tỉ lệ sai | Tỉ lệ nhóm a |", "|---|---|---|---|"]
        for b, v in s["fragmentation"].items():
            out.append(f"| {b} | {v['n']} | {pct(v['err_rate'])} % | {pct(v['a_rate'])} % |")
        out.append("")
    out += ["## Ví dụ", ""]
    seen = {}
    for r in lines:
        for p in r.pairs:
            if p.group == "ok":
                continue
            key = b_subgroup(p) if p.group == "b" else p.group
            if seen.get(key, 0) >= n_examples:
                continue
            seen[key] = seen.get(key, 0) + 1
            out.append(f"- `{key}` · {p.op}: `{p.ref_text or '∅'}` → `{p.hyp_text or '∅'}`")
    return "\n".join(out) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pred_jsonl", help="output của scripts/evaluate.py")
    ap.add_argument("--split_jsonl", help="jsonl của tập chia (lấy `document` theo `image`)")
    ap.add_argument("--out_dir", default=None)
    ap.add_argument("--dry-run", dest="dry_run", action="store_true", help="chạy trên mẫu mock")
    ap.add_argument("--no-domain-tokens", dest="domain", action="store_false",
                    help="không coi từ khóa hóa đơn/đơn vị là hợp lệ (đo độ nhạy)")
    ap.add_argument("--tokenizer_dir", default=os.path.join(HERE, "..", "data", "tokenizers"),
                    help="thư mục chứa tokenizer.json; để trống để bỏ qua phân mảnh")
    ap.add_argument("--n_boot", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n_examples", type=int, default=20)
    args = ap.parse_args()

    if args.dry_run:
        rows, source = [{"document": d, "reference": r, "predicted": h, "image": f"mock_{i}"}
                        for i, (d, r, h) in enumerate(MOCK)], "dry-run (mock)"
    elif args.pred_jsonl:
        rows, source = load_jsonl(args.pred_jsonl), os.path.basename(args.pred_jsonl)
        if args.split_jsonl:
            doc_of = {r["image"]: r.get("document", "") for r in load_jsonl(args.split_jsonl)}
            missing = [r["image"] for r in rows if r.get("image") not in doc_of]
            if missing:
                sys.exit(f"{len(missing)} dòng dự đoán không có trong {args.split_jsonl} (vd {missing[0]}) "
                         "— sai tập chia?")
            for r in rows:
                r["document"] = doc_of[r["image"]]
        else:
            from vocr.eval.significance import extract_cluster_id
            for r in rows:
                r["document"] = extract_cluster_id(r.get("image", ""), "document")
    else:
        ap.error("cần --pred_jsonl hoặc --dry-run")

    frag_fn, frag_src = load_frag_fn(args.tokenizer_dir)
    print(f"Tokenizer: {frag_src}")

    lines = [analyze_line(r["reference"], r["predicted"], args.domain,
                          document=r.get("document", ""), image=r.get("image", "")) for r in rows]
    summary = summarize(lines, args.domain, args.n_boot, args.seed, frag_fn)
    summary["source"] = source
    report = render_report(summary, lines, args.n_examples, source)
    print(report)

    if args.out_dir:
        os.makedirs(args.out_dir, exist_ok=True)
        with open(os.path.join(args.out_dir, "summary.json"), "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        with open(os.path.join(args.out_dir, "report.md"), "w", encoding="utf-8") as f:
            f.write(report)
        with open(os.path.join(args.out_dir, "lines.jsonl"), "w", encoding="utf-8") as f:
            for r in lines:
                f.write(json.dumps({
                    "image": r.image, "document": r.document, "reference": r.ref, "predicted": r.hyp,
                    "edits_raw": r.edits_raw, "edits": r.edits, "worse": r.worse,
                    "corrected_oracle": r.corrected["oracle"],
                    "pairs": [{"op": p.op, "ref": p.ref_text, "hyp": p.hyp_text,
                               "group": p.group, "detail": p.detail} for p in r.pairs if p.group != "ok"],
                }, ensure_ascii=False) + "\n")
        print(f"Đã ghi {args.out_dir}/(summary.json, report.md, lines.jsonl)")


if __name__ == "__main__":
    main()
