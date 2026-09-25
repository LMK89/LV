"""
verify_bpe_fragmentation.py — Chứng minh (ĐỘC LẬP với dataset) rằng tokenizer
byte-level BPE của Florence-2 cắt vụn âm tiết tiếng Việt thành nhiều token.

Trả lời dứt điểm câu hỏi nền của luận văn: "Florence có THẬT SỰ tách âm tiết
Việt thành nhiều token không?". Chạy trên CHÍNH 7.184 âm tiết hợp lệ trong
`vietnamese_syllables.txt` nên KHÔNG cần dataset huấn luyện.

Cách chạy:

  # (a) Dùng ĐÚNG tokenizer Florence-2 (cần transformers + tải được model):
  python src/core/nesy/verify_bpe_fragmentation.py

  # (b) OFFLINE — đưa vocab.json + merges.txt của một byte-level BPE cùng họ
  #     (vd GPT-2/BART), khi không tải được Florence:
  python src/core/nesy/verify_bpe_fragmentation.py --vocab enc.json --merges vocab.bpe

Lưu ý: GPT-2/BART/Florence-2 dùng CÙNG họ byte-level BPE (English-centric,
vocab ~50k). Merges của Florence khác GPT-2 ở vài chi tiết, nhưng hành vi cắt
vụn tiếng Việt là đại diện. Số "chuẩn" để trích vào luận văn nên lấy từ nhánh
(a) — chính tokenizer Florence-2.
"""
import argparse
import os
from collections import Counter


def load_tokenize_fn(args):
    """Trả về hàm word -> list[token]. Dùng leading space để mô phỏng từ trong câu."""
    if args.vocab and args.merges:
        from tokenizers import ByteLevelBPETokenizer
        t = ByteLevelBPETokenizer(vocab=args.vocab, merges=args.merges)
        return lambda w: t.encode(" " + w).tokens, f"vocab/merges ({os.path.basename(args.vocab)})"

    # Import muộn: nhánh offline (--vocab/--merges) ở trên KHÔNG cần
    # transformers/torch, giữ nguyên khả năng chạy trên máy không có model.
    from vocr.models.loader import load_processor
    proc = load_processor(args.model)
    tk = proc.tokenizer
    return lambda w: tk.tokenize(" " + w), args.model


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="florence-community/Florence-2-base")
    ap.add_argument("--vocab", help="vocab.json (byte-level BPE) cho chế độ offline")
    ap.add_argument("--merges", help="merges.txt/vocab.bpe cho chế độ offline")
    ap.add_argument("--syllables", default=os.path.join(here, "..", "src", "vocr", "nesy", "vietnamese_syllables.txt"))
    args = ap.parse_args()

    tokenize, src = load_tokenize_fn(args)
    syls = [l.strip() for l in open(args.syllables, encoding="utf-8") if l.strip()]

    print(f"Tokenizer : {src}")
    print(f"Âm tiết   : {len(syls)}\n")

    print("Vài ví dụ:")
    for w in ["nghiêng", "người", "trường", "được", "đường", "tiếng", "việt"]:
        toks = tokenize(w)
        print(f"  {w:>9} -> {len(toks):2d} token: {toks}")

    dist = Counter()
    for s in syls:
        k = len(tokenize(s))
        dist[k if k <= 3 else 4] += 1

    tot = len(syls)
    print("\nPhân bố độ dài token:")
    for k, label in [(1, "1-token"), (2, "2-token"), (3, "3-token"), (4, "4+-token")]:
        c = dist[k]
        print(f"  {label:>9}: {c:5d} ({c / tot * 100:5.1f}%)")
    multi = tot - dist[1]
    print(f"\n=> {multi}/{tot} = {multi / tot * 100:.1f}% âm tiết bị cắt thành 2+ token")
    print(f"=> {dist[4]}/{tot} = {dist[4] / tot * 100:.1f}% bị cắt thành 4+ token")
    print("   (đây chính là gốc rễ khiến ràng buộc cấp âm tiết khó áp ở cấp token BPE)")


if __name__ == "__main__":
    main()
