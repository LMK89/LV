"""
train.py — Fine-tune Florence-2 bằng DoRA, có/không NeSy Loss.

  λ = 0                   : DoRA thuần (đối chứng)
  λ > 0, mask_mode=rule   : NeSy Loss với mặt nạ từ luật âm tiết
  λ > 0, mask_mode=random : control — mặt nạ ngẫu nhiên CÙNG MẬT ĐỘ với mặt nạ
                            luật, để tách tác dụng "tri thức âm tiết" khỏi tác
                            dụng điều chuẩn chung chung.

Ví dụ:
  python scripts/train.py --nesy_weight 0   --seed 1 --output_dir outputs/checkpoints/lam0_s1
  python scripts/train.py --nesy_weight 0.1 --seed 1 --output_dir outputs/checkpoints/lam0.1_s1
  python scripts/train.py --nesy_weight 0.1 --mask_mode random --seed 1 --output_dir outputs/checkpoints/rand0.1_s1
"""
import argparse
import json
import logging
import os
import subprocess
import sys

import yaml
from datasets import load_dataset
from transformers import EarlyStoppingCallback, TrainingArguments, set_seed

from vocr.models.loader import load_model_and_processor, resolve_lora_config
from vocr.models.collator import make_collate_fn
from vocr.nesy.loss import NeSyTrainer

logger = logging.getLogger(__name__)


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def main():
    parser = argparse.ArgumentParser(description="DoRA fine-tuning, optional NeSy Loss")
    parser.add_argument("--train_config", type=str, default="configs/train.yaml")
    parser.add_argument("--lora_config", type=str, default="configs/dora.yaml")
    parser.add_argument("--nesy_weight", type=float, default=0.1)
    parser.add_argument("--mask_mode", type=str, default="rule", choices=["rule", "random"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--resume", type=str, default=None)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s \u2014 %(message)s",
        handlers=[logging.StreamHandler(sys.stdout),
                  logging.FileHandler(os.path.join(args.output_dir, "train.log"), encoding="utf-8")],
    )
    set_seed(args.seed)

    with open(args.train_config, "r", encoding="utf-8") as f:
        train_cfg = yaml.safe_load(f)
    lora_cfg = resolve_lora_config(args.lora_config)

    # Lưu đủ thông tin để tái lập lần chạy
    with open(os.path.join(args.output_dir, "run_info.json"), "w", encoding="utf-8") as f:
        json.dump({"args": vars(args), "git_commit": _git_commit(),
                   "train_config": train_cfg, "lora_config": lora_cfg}, f, ensure_ascii=False, indent=2)

    train_cfg["nesy_loss_weight"] = args.nesy_weight
    train_cfg["output_dir"] = args.output_dir

    model, processor, _ = load_model_and_processor(lora_cfg, train_cfg)

    logger.info(f"Loading dataset: {train_cfg['dataset_path']}")
    raw_data = load_dataset(
        "json",
        data_files={
            "train": train_cfg["dataset_path"],
            "validation": train_cfg["val_dataset_path"],
        },
    )
    logger.info(f"Train: {len(raw_data['train'])} | Val: {len(raw_data['validation'])}")

    # Drop records whose image file is missing BEFORE training starts.
    # The collator's fallback (dummy black image) exists only as a last
    # resort — silently training "black image -> text" pairs teaches the
    # model to hallucinate and corrupts results without any visible error.
    for split in ("train", "validation"):
        before = len(raw_data[split])
        raw_data[split] = raw_data[split].filter(lambda r: os.path.isfile(r.get("image", "")))
        dropped = before - len(raw_data[split])
        if dropped:
            logger.warning(
                f"[{split}] Dropped {dropped}/{before} records with missing image files. "
                "If this number is large, the dataset images were not uploaded/extracted correctly."
            )
    if len(raw_data["train"]) == 0:
        logger.error("No training records remain after filtering missing images. Aborting.")
        sys.exit(1)

    collate_fn = make_collate_fn(processor, device=None, max_length=train_cfg.get("max_label_length", 768))

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        seed=args.seed,
        data_seed=args.seed,
        per_device_train_batch_size=train_cfg["per_device_train_batch_size"],
        gradient_accumulation_steps=train_cfg["gradient_accumulation_steps"],
        learning_rate=train_cfg["learning_rate"],
        num_train_epochs=train_cfg["num_train_epochs"],
        lr_scheduler_type=train_cfg["lr_scheduler_type"],
        warmup_ratio=train_cfg.get("warmup_ratio", 0.05),
        fp16=train_cfg.get("fp16", False),
        bf16=train_cfg.get("bf16", False),
        eval_strategy=train_cfg["eval_strategy"],
        eval_steps=train_cfg["eval_steps"],
        save_strategy=train_cfg["save_strategy"],
        save_steps=train_cfg["save_steps"],
        save_total_limit=train_cfg["save_total_limit"],
        load_best_model_at_end=train_cfg["load_best_model_at_end"],
        metric_for_best_model=train_cfg["metric_for_best_model"],
        greater_is_better=train_cfg["greater_is_better"],
        logging_steps=train_cfg["logging_steps"],
        report_to=train_cfg.get("report_to", "none"),
        remove_unused_columns=False,
        label_names=["labels"],
        dataloader_num_workers=train_cfg.get("dataloader_num_workers", 0),
        dataloader_pin_memory=train_cfg.get("dataloader_pin_memory", False),
    )

    callbacks = []
    patience = train_cfg.get("early_stopping_patience", 0)
    if patience > 0:
        callbacks.append(EarlyStoppingCallback(early_stopping_patience=patience))

    trainer = NeSyTrainer(
        model=model,
        args=training_args,
        train_dataset=raw_data["train"],
        eval_dataset=raw_data["validation"],
        data_collator=collate_fn,
        callbacks=callbacks,
        nesy_weight=args.nesy_weight,
        mask_mode=args.mask_mode,
        mask_seed=args.seed,
        tokenizer=processor.tokenizer,
    )

    logger.info("=" * 60)
    logger.info("DoRA fine-tuning")
    logger.info(f"  DoRA r     : {lora_cfg['r']}")
    logger.info(f"  NeSy weight: {args.nesy_weight} (mask={args.mask_mode})")
    logger.info(f"  Seed       : {args.seed}")
    logger.info(f"  Epochs     : {train_cfg['num_train_epochs']}")
    logger.info(f"  Output     : {args.output_dir}")
    logger.info("=" * 60)

    trainer.train(resume_from_checkpoint=args.resume)

    final_path = os.path.join(args.output_dir, "final_adapter")
    os.makedirs(final_path, exist_ok=True)
    model.save_pretrained(final_path)
    processor.save_pretrained(final_path)
    logger.info(f"Adapter saved to: {final_path}")


if __name__ == "__main__":
    main()
