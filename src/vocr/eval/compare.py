"""Ghi kết quả chi tiết ra jsonl."""
import json
import logging
import os
from datetime import datetime

logger = logging.getLogger(__name__)


def save_results(records: list[dict], mode: str, output_dir: str = "outputs/results"):
    os.makedirs(output_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(output_dir, f"eval_{mode}_{ts}.jsonl")
    with open(out_path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    logger.info(f"Detailed results saved: {out_path}")
    return out_path
