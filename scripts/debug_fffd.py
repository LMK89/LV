"""Bước 1b — chẩn đoán vì sao output v1 chứa U+FFFD (199/200 dòng).

Với byte-level BPE, `tokenizer.decode(ids)` chỉ sinh U+FFFD khi DÃY BYTE của cả
chuỗi không phải UTF-8 hợp lệ. Có hai kiểu nguyên nhân:
  (1) Lỗi pipeline: decode từng token rồi nối, lệch train/suy luận (merge
      adapter, BOS/decoder start...) -> model tốt nhưng output hỏng.
  (2) Model sinh sai dãy byte (chưa học đủ) -> output hỏng thật.
Script tách hai kiểu này bằng ba lệnh con:

  tokenizer  (CPU, không cần model) round-trip nhãn, decode cả chuỗi vs từng
             token, thống kê mảnh byte.
  model      (cần checkpoint)  cấu hình BOS/decoder start, embedding có bị merge
             adapter làm đổi không, loss teacher-forcing merged vs unmerged,
             greedy decode trên ảnh TRAIN (merged vs unmerged).
  preds      (CPU) soi file eval_*.jsonl đã có.

  python scripts/debug_fffd.py tokenizer                       # tokenizer Florence-2 (cần tải HF)
  python scripts/debug_fffd.py tokenizer --gpt2_tiktoken gpt2.tiktoken   # offline, BPE GPT-2 cùng merges
  python scripts/debug_fffd.py model --checkpoint outputs/checkpoints/v1/final_adapter --n 20
  python scripts/debug_fffd.py preds --pred_jsonl outputs/results/eval_stage1_*.jsonl
"""
import argparse
import json
import os
import statistics
import unicodedata
from collections import Counter

FFFD = "�"
GPT2_PATTERN = r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


# ── Tokenizer: bọc HF (Florence-2) hoặc BPE GPT-2 (offline) chung một giao diện ──

class _HFTok:
    def __init__(self, tok):
        from transformers.models.gpt2.tokenization_gpt2 import bytes_to_unicode
        self.tok = tok
        self.byte_decoder = {v: k for k, v in bytes_to_unicode().items()}
        self.name = getattr(tok, "name_or_path", "hf")

    def encode(self, text):
        return self.tok(text, add_special_tokens=False).input_ids

    def decode(self, ids):
        return self.tok.decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)

    def token_bytes(self, tid):
        piece = self.tok.convert_ids_to_tokens(tid)
        return bytes(self.byte_decoder[c] for c in piece)


class _TiktokenTok:
    def __init__(self, path):
        import base64
        import tiktoken
        ranks = {}
        with open(path, "rb") as f:
            for line in f:
                if line.strip():
                    tok, rank = line.split()
                    ranks[base64.b64decode(tok)] = int(rank)
        self.enc = tiktoken.Encoding(name="gpt2_offline", pat_str=GPT2_PATTERN,
                                     mergeable_ranks=ranks, special_tokens={})
        self.name = f"GPT-2 BPE offline ({os.path.basename(path)})"

    def encode(self, text):
        return self.enc.encode(text)

    def decode(self, ids):
        return self.enc.decode(ids, errors="replace")

    def token_bytes(self, tid):
        return self.enc.decode_single_token_bytes(tid)


def _is_complete_utf8(b: bytes) -> bool:
    try:
        b.decode("utf-8")
        return True
    except UnicodeDecodeError:
        return False


def cmd_tokenizer(args):
    if args.gpt2_tiktoken:
        tk = _TiktokenTok(args.gpt2_tiktoken)
    elif args.model and os.path.isfile(os.path.join(args.model, "tokenizer.json")):
        from transformers import AutoTokenizer   # thư mục checkpoint: không cần torch
        tk = _HFTok(AutoTokenizer.from_pretrained(args.model))
    else:
        from vocr.models.loader import load_processor
        tk = _HFTok(load_processor(args.model).tokenizer)

    rows = load_jsonl(args.labels)
    n = len(rows)
    rt_fail, nfc_diff, full_fffd, per_tok_fffd = [], 0, 0, 0
    n_tokens = n_partial = 0
    tokens_per_line = []
    for r in rows:
        text = r["suffix"]
        if unicodedata.normalize("NFC", text) != text:
            nfc_diff += 1
        ids = tk.encode(text)
        tokens_per_line.append(len(ids))
        full = tk.decode(ids)
        if full != text:
            rt_fail.append((text, full))
        full_fffd += FFFD in full
        pieces = [tk.decode([i]) for i in ids]          # kiểu lỗi: decode từng token rồi nối
        per_tok_fffd += FFFD in "".join(pieces)
        n_tokens += len(ids)
        n_partial += sum(not _is_complete_utf8(tk.token_bytes(i)) for i in ids)

    out = [
        f"## Tokenizer: {tk.name}", "",
        f"- Nhãn: `{args.labels}` ({n} dòng, {n_tokens} token, TB {statistics.mean(tokens_per_line):.1f} token/dòng)",
        f"- Nhãn không ở dạng NFC: {nfc_diff}/{n}",
        f"- Round-trip `decode(encode(x)) == x` hỏng: **{len(rt_fail)}/{n}**",
        f"- Decode CẢ CHUỖI có U+FFFD: **{full_fffd}/{n}**",
        f"- Decode TỪNG TOKEN rồi nối có U+FFFD: **{per_tok_fffd}/{n}** ({per_tok_fffd / n:.1%})",
        f"- Token là mảnh byte (không tự thành UTF-8 hợp lệ): **{n_partial}/{n_tokens}** ({n_partial / n_tokens:.1%})",
    ]
    for text, full in rt_fail[:5]:
        out.append(f"  - round-trip lệch: `{text}` -> `{full}`")
    print("\n".join(out))


# ── Model: kiểm lệch train/suy luận trên ảnh TRAIN ─────────────────────────────

def _cfg_value(obj, name):
    for o in (obj, getattr(obj, "text_config", None)):
        if o is not None and getattr(o, name, None) is not None:
            return getattr(o, name)
    return None


def _embedding_snapshot(model, vocab_rows):
    """Clone mọi nn.Embedding có >= vocab_rows hàng (embedding token của decoder/encoder)."""
    import torch
    return {name: m.weight.detach().clone() for name, m in model.named_modules()
            if isinstance(m, torch.nn.Embedding) and m.num_embeddings >= vocab_rows}


def _teacher_forced(model, batch):
    import torch
    with torch.no_grad():
        out = model(**batch)
    labels = batch["labels"]
    mask = labels != -100
    pred = out.logits.argmax(-1)
    acc = ((pred == labels) & mask).sum().item() / max(mask.sum().item(), 1)
    return out.loss.item(), acc, pred


def cmd_model(args):
    import torch
    from PIL import Image

    from vocr.eval.metrics import corpus_cer
    from vocr.models.collator import make_collate_fn
    from vocr.models.loader import load_base_model, load_lora_model

    rows = load_jsonl(args.jsonl)
    if args.only_v1_train:   # ảnh checkpoint v1 THỰC SỰ đã thấy lúc train
        seen = {r["image"] for r in load_jsonl(args.raw_labels) if r.get("split_v1") == "train"}
        rows = [r for r in rows if r["image"] in seen]
    rows = rows[: args.n]
    dev = args.device
    report = [f"## Model: `{args.checkpoint}` — {len(rows)} ảnh từ `{args.jsonl}`", ""]

    unmerged, processor = load_lora_model(args.checkpoint, dev, merge=False)
    tok = processor.tokenizer
    vocab_rows = len(tok)

    # 1) Token đặc biệt + cấu hình sinh
    base_cfg = unmerged.get_base_model().config if hasattr(unmerged, "get_base_model") else unmerged.config
    gen_cfg = getattr(unmerged, "generation_config", None)
    lab = tok(rows[0]["suffix"]).input_ids
    report += ["### 1. Token đặc biệt và cấu hình sinh", "",
               "| Mục | Giá trị |", "|---|---|",
               f"| tokenizer bos/eos/pad | {tok.bos_token_id} / {tok.eos_token_id} / {tok.pad_token_id} |",
               f"| len(tokenizer) | {vocab_rows} |",
               f"| nhãn: 3 id đầu / id cuối | {lab[:3]} / {lab[-1]} |"]
    for key in ("decoder_start_token_id", "forced_bos_token_id", "forced_eos_token_id",
                "bos_token_id", "eos_token_id", "tie_word_embeddings"):
        report.append(f"| config.{key} | {_cfg_value(base_cfg, key)} |")
    if gen_cfg is not None:
        for key in ("decoder_start_token_id", "forced_bos_token_id", "forced_eos_token_id",
                    "num_beams", "no_repeat_ngram_size"):
            report.append(f"| generation_config.{key} | {getattr(gen_cfg, key, None)} |")
    report.append("")

    # 2) merge_and_unload có làm đổi embedding đầu vào không (lm_head buộc với embedding)
    base_id = args.base_model
    adapter_cfg = os.path.join(args.checkpoint, "adapter_config.json")
    if base_id is None and os.path.isfile(adapter_cfg):
        with open(adapter_cfg, encoding="utf-8") as f:
            base_id = json.load(f).get("base_model_name_or_path")
    base = load_base_model(base_id, dtype=torch.float32)
    before = _embedding_snapshot(base, vocab_rows)
    del base
    merged, _ = load_lora_model(args.checkpoint, dev, merge=True)
    after = _embedding_snapshot(merged, vocab_rows)
    report += ["### 2. Embedding trước/sau `merge_and_unload()`", "",
               "| Embedding | max |Δ| | số hàng đổi |", "|---|---|---|"]
    for name, w0 in before.items():
        w1 = after.get(name)
        if w1 is None or w1.shape != w0.shape:
            report.append(f"| `{name}` | (không khớp tên/shape sau merge) | |")
            continue
        diff = (w1.cpu() - w0.cpu()).abs()
        report.append(f"| `{name}` | {diff.max().item():.3e} | {(diff.max(-1).values > 1e-6).sum().item()} |")
    report.append("")

    # 3) Teacher forcing: merged vs unmerged trên cùng batch (chính collator lúc train)
    collate = make_collate_fn(processor, device=dev)
    report += ["### 3. Teacher forcing trên ảnh train (collator lúc train)", "",
               "| Model | loss TB | acc token (argmax) | argmax decode có U+FFFD |", "|---|---|---|---|"]
    for label, m in (("unmerged", unmerged), ("merged", merged)):
        losses, accs, fffd = [], [], 0
        for r in rows:
            batch = collate([r])
            loss, acc, pred = _teacher_forced(m, batch)
            ids = [i for i, l in zip(pred[0].tolist(), batch["labels"][0].tolist()) if l != -100]
            txt = tok.decode(ids, skip_special_tokens=True)
            losses.append(loss); accs.append(acc); fffd += FFFD in txt
        report.append(f"| {label} | {statistics.mean(losses):.3f} | {statistics.mean(accs):.1%} | {fffd}/{len(rows)} |")
    report.append("")

    # 4) Sinh tự do (greedy, không ràng buộc) — merged vs unmerged; decode cả chuỗi vs từng token
    report += ["### 4. Sinh tự do trên ảnh train (greedy, không ràng buộc)", "",
               "| Model | CER corpus | U+FFFD (decode cả chuỗi) | U+FFFD (decode từng token) |", "|---|---|---|---|"]
    samples = []
    for label, m in (("unmerged", unmerged), ("merged", merged)):
        # adapter đã được chèn vào module của base model: gọi generate của base để
        # tránh lớp bọc PeftModelForCausalLM (task_type CAUSAL_LM cho model seq2seq)
        gm = m.get_base_model() if hasattr(m, "get_base_model") else m
        preds, fffd_full, fffd_tok = [], 0, 0
        for r in rows:
            image = Image.open(r["image"]).convert("RGB")
            prompt = r.get("prefix", "<OCR>")
            # cùng tham số processor như scripts/evaluate.py
            inputs = processor(text=prompt, images=image, return_tensors="pt",
                               do_resize=True, resample=3).to(dev)
            with torch.no_grad():
                gen = gm.generate(**inputs, max_new_tokens=args.max_new_tokens, num_beams=1,
                                 use_cache=False)[0].tolist()
            pred = processor.batch_decode([gen], skip_special_tokens=True)[0].replace(prompt, "").strip()
            per_tok = "".join(tok.decode([i], skip_special_tokens=True) for i in gen)
            preds.append(pred); fffd_full += FFFD in pred; fffd_tok += FFFD in per_tok
            if len(preds) <= 3:
                samples.append((label, r["suffix"], pred, gen[:8]))
        refs = [r["suffix"] for r in rows]
        report.append(f"| {label} | {corpus_cer(refs, preds):.1%} | {fffd_full}/{len(rows)} | {fffd_tok}/{len(rows)} |")
    report += ["", "Ví dụ: nhãn → dự đoán, 8 id sinh đầu tiên", ""]
    for label, ref, pred, head in samples:
        report.append(f"- [{label}] `{ref}` → `{pred}` — `{head}`")

    text = "\n".join(report) + "\n"
    print(text)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(text)


def cmd_preds(args):
    rows = load_jsonl(args.pred_jsonl)
    preds = [r["predicted"] for r in rows]
    n = len(rows)
    with_fffd = [p for p in preds if FFFD in p]
    chars = Counter()
    for p in with_fffd:
        for i, c in enumerate(p):
            if c == FFFD:
                chars[p[i - 1] if i else "^"] += 1
    ratio = [p.count(FFFD) / max(len(p), 1) for p in with_fffd]
    print("\n".join([
        f"## Dự đoán: `{args.pred_jsonl}`", "",
        f"- Dòng có U+FFFD: **{len(with_fffd)}/{n}**; số dự đoán khác nhau: {len(set(preds))}/{n}",
        f"- Tỉ lệ ký tự U+FFFD trong dòng hỏng: TB {statistics.mean(ratio) if ratio else 0:.1%}",
        f"- Ký tự đứng ngay trước U+FFFD (nhiều nhất): {chars.most_common(10)}",
        *[f"  - `{r['reference']}` → `{r['predicted']}`" for r in rows if FFFD in r["predicted"]][:5],
    ]))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    t = sub.add_parser("tokenizer")
    t.add_argument("--labels", default="data/splits/train.jsonl")
    t.add_argument("--model", default=None, help="id Florence-2, hoặc thư mục checkpoint có tokenizer.json (chạy không cần torch)")
    t.add_argument("--gpt2_tiktoken", default=None, help="file gpt2.tiktoken để chạy offline")
    t.set_defaults(func=cmd_tokenizer)

    m = sub.add_parser("model")
    m.add_argument("--checkpoint", required=True)
    m.add_argument("--base_model", default=None)
    m.add_argument("--jsonl", default="data/splits/train.jsonl")
    m.add_argument("--n", type=int, default=20)
    m.add_argument("--only_v1_train", action="store_true",
                   help="chỉ lấy ảnh thuộc train của v1 (khi soi checkpoint v1)")
    m.add_argument("--raw_labels", default="data/raw/labels.jsonl")
    m.add_argument("--max_new_tokens", type=int, default=128)
    m.add_argument("--device", default="cuda" if _has_cuda() else "cpu")
    m.add_argument("--out", default=None, help="ghi báo cáo markdown ra file")
    m.set_defaults(func=cmd_model)

    p = sub.add_parser("preds")
    p.add_argument("--pred_jsonl", required=True)
    p.set_defaults(func=cmd_preds)

    args = ap.parse_args()
    args.func(args)


def _has_cuda():
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


if __name__ == "__main__":
    main()
