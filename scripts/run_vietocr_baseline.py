"""Chạy baseline VietOCR thật trên một tập jsonl.

Khác bản v1: KHÔNG còn chế độ "simulation". Thiếu thư viện `vietocr` hoặc thiếu
trọng số thì dừng hẳn — không bao giờ sinh dự đoán giả từ nhãn.
"""
import argparse
import json
import os
import sys
import time

from PIL import Image
from tqdm import tqdm

from vocr.eval.compare import save_results
from vocr.eval.metrics import calculate_cer, calculate_wer, normalized_edit_similarity


def main():
    parser = argparse.ArgumentParser(description="Evaluate VietOCR baseline")
    parser.add_argument("--test_jsonl", type=str, default="data/splits/test.jsonl")
    parser.add_argument("--config_path", type=str, default="weights/config.yml",
                        help="config VietOCR tải sẵn (chạy offline)")
    parser.add_argument("--weights_path", type=str, default="weights/vgg_transformer.pth")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--output_dir", type=str, default="outputs/results")
    args = parser.parse_args()

    try:
        from vietocr.tool.config import Cfg
        from vietocr.tool.predictor import Predictor
    except ImportError:
        sys.exit("Thiếu thư viện vietocr: pip install vietocr")
    for p in (args.config_path, args.weights_path):
        if not os.path.isfile(p):
            sys.exit(f"Không tìm thấy {p}")

    config = Cfg.load_config_from_file(args.config_path)
    config["weights"] = args.weights_path
    config["device"] = args.device
    detector = Predictor(config)

    with open(args.test_jsonl, "r", encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]
    print(f"Loaded {len(records)} records from {args.test_jsonl}")

    detailed, latencies = [], []
    for rec in tqdm(records, desc="VietOCR"):
        img_path, ref_text = rec["image"], rec["suffix"]
        image = Image.open(img_path).convert("RGB")
        start = time.perf_counter()
        pred_text = detector.predict(image)
        latency = time.perf_counter() - start
        latencies.append(latency)
        detailed.append({
            "image": img_path, "reference": ref_text, "predicted": pred_text,
            "cer": round(calculate_cer(ref_text, pred_text), 4),
            "wer": round(calculate_wer(ref_text, pred_text), 4),
            "edit_sim": round(normalized_edit_similarity(ref_text, pred_text), 4),
            "latency_s": round(latency, 4),
        })

    n = len(detailed)
    print(f"N={n} | CER={sum(r['cer'] for r in detailed) / n:.4f} "
          f"| WER={sum(r['wer'] for r in detailed) / n:.4f} "
          f"| latency={sum(latencies) / n * 1000:.1f} ms/ảnh")
    print("Saved:", save_results(detailed, "VietOCR", output_dir=args.output_dir))


if __name__ == "__main__":
    main()
