"""Chia train/val/test theo NHÓM để không rò rỉ giữa các tập.

Tên ảnh có dạng  <ngày>_<stt>_<mã bài>_<phần>_tg_<đoạn>_<dòng>.png, ví dụ
20160527_0173_25432_1_tg_0_0.png. Cùng một bài báo (<mã bài>) được nhiều tài
liệu khác nhau chép lại, nên chia theo tài liệu vẫn để lọt câu trùng nguyên văn
giữa train và test. Mặc định vì vậy chia theo <mã bài> (--group article).

Thuật toán: xáo các nhóm theo seed, rồi gán lần lượt mỗi nhóm vào tập đang
thiếu nhiều nhất so với tỉ lệ mục tiêu (tính theo số dòng). Vì các nhóm lớn
nhỏ khác nhau, thử --n_seeds seed và chọn seed có điểm thấp nhất theo luật CỐ
ĐỊNH (chỉ dựa trên thống kê nhóm, không nhìn kết quả mô hình):
    điểm = Σ_tập |tỉ lệ thực − tỉ lệ mục tiêu| + max_{val,test} (tỉ trọng nhóm lớn nhất)

  python scripts/split_data.py            # ghi data/splits/{train,val,test}.jsonl + split_report.md
"""
import argparse
import json
import os
import random
from collections import defaultdict

SPLITS = ("train", "val", "test")


def parse_name(image_path: str) -> dict:
    stem = os.path.splitext(os.path.basename(image_path))[0]
    head, _, _ = stem.partition("_tg_")
    parts = head.split("_")
    return {"document": head, "article": parts[2]}


def assign_groups(sizes: dict, ratios: dict, seed: int) -> dict:
    groups = sorted(sizes)
    random.Random(seed).shuffle(groups)
    total = sum(sizes.values())
    filled = {s: 0 for s in SPLITS}
    assignment = {}
    for g in groups:
        # tập thiếu nhiều nhất (theo tỉ lệ) nhận nhóm này
        split = max(SPLITS, key=lambda s: ratios[s] * total - filled[s])
        assignment[g] = split
        filled[split] += sizes[g]
    return assignment


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--labels", default="data/raw/labels.jsonl")
    ap.add_argument("--out_dir", default="data/splits")
    ap.add_argument("--group", default="article", choices=["article", "document"])
    ap.add_argument("--ratios", default="0.7,0.15,0.15", help="train,val,test")
    ap.add_argument("--n_seeds", type=int, default=200, help="số seed thử (0..n_seeds-1)")
    args = ap.parse_args()

    ratios = dict(zip(SPLITS, (float(x) for x in args.ratios.split(","))))
    with open(args.labels, encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]

    sizes = defaultdict(int)
    for r in records:
        r.update(parse_name(r["image"]))
        sizes[r[args.group]] += 1
    total = len(records)

    def score(assign):
        filled = {sp: 0 for sp in SPLITS}
        biggest = {sp: 0 for sp in SPLITS}
        for g, sp in assign.items():
            filled[sp] += sizes[g]
            biggest[sp] = max(biggest[sp], sizes[g])
        if min(filled.values()) == 0:
            return float("inf")
        dev = sum(abs(filled[sp] / total - ratios[sp]) for sp in SPLITS)
        dom = max(biggest[sp] / filled[sp] for sp in ("val", "test"))
        return dev + dom

    seed = min(range(args.n_seeds), key=lambda sd: score(assign_groups(sizes, ratios, sd)))
    assignment = assign_groups(sizes, ratios, seed)

    os.makedirs(args.out_dir, exist_ok=True)
    by_split = {s: [r for r in records if assignment[r[args.group]] == s] for s in SPLITS}
    for s, rows in by_split.items():
        with open(os.path.join(args.out_dir, f"{s}.jsonl"), "w", encoding="utf-8") as f:
            for r in rows:
                out = {"image": r["image"], "prefix": r.get("prefix", "<OCR>"), "suffix": r["suffix"],
                       "document": r["document"], "article": r["article"]}
                f.write(json.dumps(out, ensure_ascii=False) + "\n")

    # Báo cáo + kiểm tra rò rỉ
    def keyset(s, k):
        return {r[k] for r in by_split[s]}
    lines = [
        "# Báo cáo chia dữ liệu", "",
        f"- Nhãn: `{args.labels}` ({len(records)} dòng)",
        f"- Khóa nhóm: `{args.group}` · tỉ lệ mục tiêu {args.ratios}",
        f"- Seed chọn: {seed} (tốt nhất trong 0..{args.n_seeds - 1}, điểm {score(assignment):.3f})", "",
        "| Tập | Số dòng | Tỉ lệ | Số bài | Số tài liệu | Bài lớn nhất |", "|---|---|---|---|---|---|",
    ]
    for s in SPLITS:
        n = len(by_split[s])
        top = max(keyset(s, args.group), key=lambda g: sizes[g])
        lines.append(f"| {s} | {n} | {n / len(records):.1%} | {len(keyset(s, 'article'))} "
                     f"| {len(keyset(s, 'document'))} | {top} ({sizes[top] / n:.0%}) |")
    lines += ["", "## Kiểm tra rò rỉ (phải bằng 0)", "", "| Cặp | Bài chung | Tài liệu chung | Câu trùng nguyên văn |", "|---|---|---|---|"]
    leaks = 0
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        row = [len(keyset(a, k) & keyset(b, k)) for k in ("article", "document", "suffix")]
        leaks += sum(row)
        lines.append(f"| {a} ∩ {b} | " + " | ".join(map(str, row)) + " |")
    report = "\n".join(lines) + "\n"
    with open(os.path.join(args.out_dir, "split_report.md"), "w", encoding="utf-8") as f:
        f.write(report)
    print(report)
    if leaks:
        raise SystemExit(f"Rò rỉ: {leaks} phần tử chung giữa các tập")


if __name__ == "__main__":
    main()
