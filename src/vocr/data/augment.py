"""
augment.py — Augmentation ảnh (STEP 1 pipeline).

Đọc config YAML (input_jsonl, output_jsonl, output_image_dir, augment_multiplier,
pipeline[]), áp albumentations theo `pipeline`, ghi ảnh augmented + jsonl mới
(gồm bản GỐC + N bản augmented mỗi ảnh).

Chạy: python -m vocr.data.augment --config configs/augment.yaml

Phần dễ vỡ nhất: TÊN THAM SỐ albumentations đổi giữa 1.x và 2.x. Builder dưới
đây map theo API 2.x và bọc try/except để một transform lỗi không làm sập cả run.
"""
import argparse
import json
import logging
import math
import os

import yaml

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def _build_one(name: str, p: float, params: dict):
    """Dựng một transform albumentations (API 2.x) từ {name, p, params}."""
    import albumentations as A
    try:
        if name == "motion_blur":
            return A.MotionBlur(blur_limit=tuple(params.get("blur_limit", [3, 7])), p=p)
        if name == "blur":
            return A.Blur(blur_limit=tuple(params.get("blur_limit", [3, 7])), p=p)
        if name == "gaussian_noise":
            # 2.x: dùng std_range (tỉ lệ 0..1) thay var_limit (0..255^2).
            vl = params.get("var_limit", [10.0, 50.0])
            std = (math.sqrt(vl[0]) / 255.0, math.sqrt(vl[1]) / 255.0)
            return A.GaussNoise(std_range=std, p=p)
        if name == "perspective":
            return A.Perspective(scale=tuple(params.get("scale", [0.05, 0.15])), p=p)
        if name == "random_shadow":
            lo = params.get("num_shadows_lower", 1)
            hi = params.get("num_shadows_upper", 2)
            return A.RandomShadow(
                shadow_roi=tuple(params.get("shadow_roi", [0, 0.5, 1, 1])),
                num_shadows_limit=(lo, hi), p=p)
        if name == "random_brightness_contrast":
            return A.RandomBrightnessContrast(
                brightness_limit=params.get("brightness_limit", 0.2),
                contrast_limit=params.get("contrast_limit", 0.2), p=p)
        if name == "rotate":
            lim = params.get("limit", [-10, 10])
            return A.Rotate(limit=tuple(lim) if isinstance(lim, list) else lim, p=p)
    except Exception as e:
        logger.warning("Bỏ transform %r (lỗi khởi tạo: %s)", name, e)
        return None
    logger.warning("Không biết transform %r — bỏ qua.", name)
    return None


def build_transform(pipeline_cfg: list, seed: int = 42):
    import albumentations as A
    tfs = [
        t for t in (_build_one(e["name"], e.get("p", 0.5), e.get("params", {}))
                    for e in pipeline_cfg)
        if t is not None
    ]
    try:
        return A.Compose(tfs, seed=seed)
    except TypeError:
        return A.Compose(tfs)


def main():
    import cv2
    ap = argparse.ArgumentParser(description="Augment ảnh theo config YAML")
    ap.add_argument("--config", required=True)
    args = ap.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    in_jsonl = cfg["input_jsonl"]
    out_jsonl = cfg["output_jsonl"]
    out_dir = cfg["output_image_dir"]
    mult = int(cfg.get("augment_multiplier", 1))
    seed = int(cfg.get("seed", 42))

    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)

    os.makedirs(out_dir, exist_ok=True)
    transform = build_transform(cfg.get("pipeline", []), seed=seed)

    records = []
    with open(in_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    logger.info("Đọc %d bản ghi từ %s", len(records), in_jsonl)

    out = []
    n_aug = 0
    n_skipped = 0
    for rec in records:
        img_path = rec.get("image", "")
        img = cv2.imread(img_path)
        if img is None:
            n_skipped += 1
            logger.warning("Bỏ qua bản ghi (ảnh thiếu/không đọc được): %s", img_path)
            continue
        out.append(rec)  # luôn giữ bản gốc hợp lệ
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        stem = os.path.splitext(os.path.basename(img_path))[0]
        for i in range(mult):
            aug = transform(image=img)["image"]
            out_path = os.path.join(out_dir, f"{stem}_aug{i}.jpg")
            cv2.imwrite(out_path, cv2.cvtColor(aug, cv2.COLOR_RGB2BGR))
            new = dict(rec)
            new["image"] = out_path
            out.append(new)
            n_aug += 1

    os.makedirs(os.path.dirname(out_jsonl) or ".", exist_ok=True)
    with open(out_jsonl, "w", encoding="utf-8") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    logger.info(
        "Hoàn tất: Đọc %d, Bỏ qua %d lỗi ảnh -> Ghi %d bản ghi (gốc + %d augmented) -> %s",
        len(records), n_skipped, len(out), n_aug, out_jsonl
    )


if __name__ == "__main__":
    main()
