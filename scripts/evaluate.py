"""Đánh giá MỘT cấu hình (model × ràng buộc decode) trên một tập jsonl.

Khác bản v1:
- Mỗi lần chạy là một cấu hình, đặt tên bằng --name (thay cho ma trận 6 biến thể
  cố định) — tiện cho quét λ × seed.
- Bỏ vòng "retry với hint": Florence-2 chỉ nhận task token (<OCR>), đưa câu chỉ
  dẫn tiếng Việt vào prompt là input ngoài phân phối.
- Ghi cả CER/WER mức corpus (micro) bên cạnh trung bình theo mẫu (macro).

Ví dụ:
  python scripts/evaluate.py --name base
  python scripts/evaluate.py --name base+fsm --constraint automaton
  python scripts/evaluate.py --name lam0_s1 --checkpoint outputs/checkpoints/lam0_s1/final_adapter
"""
import argparse
import json
import logging
import os
import statistics
import sys
import time
from datetime import datetime

import torch
from PIL import Image
from tqdm import tqdm

from vocr.eval.metrics import (
    calculate_cer, calculate_wer, corpus_cer, corpus_wer, normalized_edit_similarity,
)
from vocr.models.loader import DEFAULT_BASE_MODEL_ID, load_base_model, load_lora_model, load_processor

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def load_jsonl(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


@torch.no_grad()
def run_inference(model, processor, image, prompt, max_new_tokens, num_beams, device, logits_processors):
    max_len = getattr(processor.tokenizer, "model_max_length", 1024) or 1024
    inputs = processor(
        text=prompt, images=image, return_tensors="pt", do_resize=True, resample=3,
        truncation=True, max_length=max_len,
    ).to(device)
    start = time.perf_counter()
    generated_ids = model.generate(
        **inputs,
        logits_processor=logits_processors,
        max_new_tokens=max_new_tokens,
        num_beams=num_beams,
        use_cache=False,
        early_stopping=False,
    )
    latency = time.perf_counter() - start
    raw = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return raw.replace(prompt, "").strip(), latency


def main():
    parser = argparse.ArgumentParser(description="Evaluate one configuration")
    parser.add_argument("--name", type=str, required=True, help="tên cấu hình, dùng đặt tên file kết quả")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="adapter đã train; bỏ trống = Florence-2 gốc (zero-shot)")
    parser.add_argument("--base_model", type=str, default=DEFAULT_BASE_MODEL_ID)
    parser.add_argument("--test_jsonl", type=str, default="data/splits/test.jsonl")
    parser.add_argument("--constraint", type=str, default="none", choices=["none", "token", "automaton"],
                        help="ràng buộc âm tiết lúc decode")
    parser.add_argument("--constraint_top_k", type=int, default=50)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--num_beams", type=int, default=3)
    parser.add_argument("--output_dir", type=str, default="outputs/results")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    records = load_jsonl(args.test_jsonl)
    if not records:
        sys.exit(f"Tập test rỗng: {args.test_jsonl}")

    if args.checkpoint:
        model, processor = load_lora_model(args.checkpoint, args.device)
    else:
        processor = load_processor(args.base_model)
        model = load_base_model(args.base_model, dtype=torch.float32, device=args.device)
    model.eval()

    logits_processors = []
    if args.constraint != "none":
        from vocr.nesy.constraints import VietnamesePhonologyLogitsProcessor
        logits_processors.append(VietnamesePhonologyLogitsProcessor(
            tokenizer=processor.tokenizer, top_k=args.constraint_top_k, mode=args.constraint))

    os.makedirs(args.output_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(args.output_dir, f"eval_{args.name}_{ts}.jsonl")

    detailed = []
    with open(out_path, "w", encoding="utf-8") as fout:
        for rec in tqdm(records, desc=f"Evaluating [{args.name}]", unit="img"):
            image = Image.open(rec["image"]).convert("RGB")
            ref = rec["suffix"]
            pred, lat = run_inference(model, processor, image, rec.get("prefix", "<OCR>"),
                                      args.max_new_tokens, args.num_beams, args.device, logits_processors)
            row = {
                "image": rec["image"], "reference": ref, "predicted": pred,
                "cer": round(calculate_cer(ref, pred), 4), "wer": round(calculate_wer(ref, pred), 4),
                "edit_sim": round(normalized_edit_similarity(ref, pred), 4),
                "latency_s": round(lat, 4),
            }
            detailed.append(row)
            # ghi từng dòng: chết giữa chừng vẫn giữ phần đã chạy
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")
            fout.flush()

    refs = [r["reference"] for r in detailed]
    preds = [r["predicted"] for r in detailed]
    lats = sorted(r["latency_s"] for r in detailed)
    summary = {
        "name": args.name, "checkpoint": args.checkpoint, "constraint": args.constraint,
        "test_jsonl": args.test_jsonl, "n": len(detailed),
        "corpus_cer": round(corpus_cer(refs, preds), 4),
        "corpus_wer": round(corpus_wer(refs, preds), 4),
        "mean_cer": round(statistics.mean(r["cer"] for r in detailed), 4),
        "median_cer": round(statistics.median(r["cer"] for r in detailed), 4),
        "mean_wer": round(statistics.mean(r["wer"] for r in detailed), 4),
        "n_replacement_char": sum("�" in p for p in preds),
        "n_unique_pred": len(set(preds)),
        "latency_mean_s": round(statistics.mean(lats), 4),
        "latency_p95_s": lats[min(len(lats) - 1, int(len(lats) * 0.95))],
        "output_path": out_path,
    }
    with open(out_path.replace(".jsonl", "_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    logger.info(json.dumps(summary, ensure_ascii=False, indent=2))
    # Chuông báo động kiểu lỗi v1 (output cố định / UTF-8 hỏng)
    if summary["n_unique_pred"] < 0.5 * summary["n"] or summary["n_replacement_char"] > 0.1 * summary["n"]:
        logger.warning("Output bất thường: ít dự đoán khác nhau hoặc nhiều ký tự U+FFFD — kiểm tra pipeline trước khi tin số liệu.")


if __name__ == "__main__":
    main()
