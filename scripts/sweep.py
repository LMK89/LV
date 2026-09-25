"""
sweep.py — Điều phối ma trận thí nghiệm Bước 3: Quét λ × mask × seed (15 lượt).

Hỗ trợ các chế độ:
  --plan    : In ra kế hoạch 15 runs theo thứ tự seed-major.
  --status  : Báo cáo tiến độ hoàn thành các lượt từ thư mục output.
  --dry-run : Chạy kiểm tra nhanh pipeline trên CPU với cấu hình nhỏ gọn.
  --run     : Thực thi toàn bộ (hoặc các lượt còn lại) của sweep trên GPU.
  --only    : Chỉ chạy duy nhất một run_id cụ thể.
  --force   : Tiếp tục chạy ngay cả khi git working tree chưa clean.
"""
import argparse
import csv
import json
import logging
import os
import subprocess
import sys
import time
from pathlib import Path

import yaml

if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def _is_git_clean() -> bool:
    try:
        status = subprocess.check_output(["git", "status", "--porcelain"], text=True).strip()
        return len(status) == 0
    except Exception:
        return False


def load_sweep_config(config_path: str = "configs/sweep_step3.yaml") -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_run_list(sweep_cfg: dict) -> list[dict]:
    seeds = sweep_cfg.get("seeds", [1, 2, 3])
    conditions = sweep_cfg.get("conditions", [])
    order = sweep_cfg.get("order", "seed_major")

    runs = []
    if order == "seed_major":
        for seed in seeds:
            for cond in conditions:
                cond_id = cond["id"]
                run_id = f"{cond_id}_s{seed}"
                runs.append({
                    "run_id": run_id,
                    "condition": cond_id,
                    "seed": seed,
                    "nesy_weight": cond["nesy_weight"],
                    "mask_mode": cond["mask_mode"],
                })
    else:
        for cond in conditions:
            for seed in seeds:
                cond_id = cond["id"]
                run_id = f"{cond_id}_s{seed}"
                runs.append({
                    "run_id": run_id,
                    "condition": cond_id,
                    "seed": seed,
                    "nesy_weight": cond["nesy_weight"],
                    "mask_mode": cond["mask_mode"],
                })
    return runs


def print_plan(runs: list[dict]):
    print(f"\n{'='*75}")
    print(f"KẾ HOẠCH SWEEP BƯỚC 3: QUÉT λ × MASK × SEED ({len(runs)} RUNS)")
    print(f"{'='*75}")
    print(f"{'STT':<5} {'Run ID':<20} {'Condition':<15} {'Seed':<6} {'λ (weight)':<12} {'Mask Mode':<10}")
    print(f"{'-'*75}")
    for idx, r in enumerate(runs, 1):
        print(f"{idx:<5} {r['run_id']:<20} {r['condition']:<15} {r['seed']:<6} {r['nesy_weight']:<12} {r['mask_mode']:<10}")
    print(f"{'='*75}\n")


def check_status(sweep_cfg: dict, runs: list[dict]):
    out_root = Path(sweep_cfg.get("output_root", "outputs/step3"))
    print(f"\n{'='*75}")
    print(f"TIẾN ĐỘ THỰC THI SWEEP BƯỚC 3 ({out_root})")
    print(f"{'='*75}")
    print(f"{'Run ID':<20} {'Status':<15} {'Best Step':<12} {'Best eval_ce':<15}")
    print(f"{'-'*75}")

    done_count = 0
    for r in runs:
        run_dir = out_root / r["run_id"]
        done_file = run_dir / "DONE"
        run_info = run_dir / "run_info.json"
        
        status = "PENDING"
        best_step = "-"
        best_ce = "-"

        if done_file.exists():
            status = "DONE"
            done_count += 1
        elif (run_dir / "train.log").exists():
            status = "RUNNING/FAILED"

        if run_info.exists():
            try:
                with open(run_info, "r", encoding="utf-8") as f:
                    info = json.load(f)
                    best_step = str(info.get("best_step", "-"))
                    best_ce = f"{info.get('best_eval_ce', '-'):.4f}" if isinstance(info.get("best_eval_ce"), float) else "-"
            except Exception:
                pass

        print(f"{r['run_id']:<20} {status:<15} {best_step:<12} {best_ce:<15}")

    print(f"{'='*75}")
    print(f"Tổng kết: Đã hoàn thành {done_count}/{len(runs)} lượt chạy.\n")


def _update_manifest(manifest_path: Path, row_dict: dict):
    headers = [
        "run_id", "condition", "seed", "nesy_weight", "mask_mode", "git_commit",
        "status", "wall_min", "best_step", "best_eval_ce", "val_corpus_cer",
        "val_fffd_lines", "byte_acc", "ascii_acc", "acc_gap", "delta_oracle"
    ]
    file_exists = manifest_path.exists()
    rows = []
    updated = False

    if file_exists:
        try:
            with open(manifest_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for r in reader:
                    if r.get("run_id") == row_dict["run_id"]:
                        rows.append(row_dict)
                        updated = True
                    else:
                        rows.append(r)
        except Exception:
            pass

    if not updated:
        rows.append(row_dict)

    with open(manifest_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        for r in rows:
            clean_row = {k: r.get(k, "-") for k in headers}
            writer.writerow(clean_row)


def run_sweep(sweep_cfg: dict, runs: list[dict], dry_run: bool = False, only: str = None, force: bool = False):
    out_root = Path(sweep_cfg.get("output_root", "outputs/step3"))
    out_root.mkdir(parents=True, exist_ok=True)
    manifest_path = out_root / "manifest.csv"

    if dry_run:
        os.environ["DRY_RUN"] = "1"

    train_config = "configs/train_dryrun.yaml" if dry_run else sweep_cfg.get("train_config", "configs/train.yaml")
    lora_config = sweep_cfg.get("lora_config", "configs/dora.yaml")

    if not dry_run and not _is_git_clean():
        msg = "Git working tree chưa clean! Để đảm bảo tính tái lập học thuật, cần commit trước khi chạy sweep."
        if not force:
            logger.error(msg + " Sử dụng --force nếu bạn chắc chắn muốn bỏ qua kiểm tra này.")
            sys.exit(1)
        else:
            logger.warning("CẢNH BÁO (--force được bật): " + msg)

    target_runs = [r for r in runs if only is None or r["run_id"] == only]
    logger.info(f"Bắt đầu điều phối sweep ({len(target_runs)} lượt, dry_run={dry_run})...")

    for idx, r in enumerate(target_runs, 1):
        run_id = r["run_id"]
        run_dir = out_root / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        done_file = run_dir / "DONE"

        if done_file.exists():
            logger.info(f"[{idx}/{len(target_runs)}] {run_id} ĐÃ HOÀN THÀNH. Bỏ qua.")
            continue

        logger.info(f"\n{'='*70}\n[{idx}/{len(target_runs)}] BẮT ĐẦU: {run_id}\n{'='*70}")
        start_time = time.time()

        # Kiểm tra checkpoint dở dang để tự động resume
        resume_ckpt = None
        checkpoints = sorted(run_dir.glob("checkpoint-*"), key=os.path.getmtime)
        if checkpoints:
            resume_ckpt = str(checkpoints[-1])
            logger.info(f"Tìm thấy checkpoint dở dang: {resume_ckpt}. Tự động resume...")

        # ── PHA 1: Huấn luyện (Train) ──────────────────────────────────
        train_cmd = [
            sys.executable, "scripts/train.py",
            "--train_config", train_config,
            "--lora_config", lora_config,
            "--nesy_weight", str(r["nesy_weight"]),
            "--mask_mode", r["mask_mode"],
            "--seed", str(r["seed"]),
            "--output_dir", str(run_dir),
        ]
        if resume_ckpt:
            train_cmd.extend(["--resume", resume_ckpt])

        ret = subprocess.run(train_cmd)
        if ret.returncode != 0:
            logger.error(f"Lượt {run_id} THẤT BẠI ở pha Train với mã lỗi {ret.returncode}.")
            sys.exit(ret.returncode)

        # ── PHA 2: Đánh giá trên tập Val (Evaluate) ────────────────────
        final_adapter = run_dir / "final_adapter"
        val_eval_file = run_dir / "eval_val.jsonl"
        val_split = sweep_cfg.get("eval", {}).get("split_jsonl", "data/splits/val.jsonl")
        num_beams = sweep_cfg.get("eval", {}).get("num_beams", 3)
        max_tokens = sweep_cfg.get("eval", {}).get("max_new_tokens", 256)

        val_cer = "-"
        val_fffd = "-"
        byte_acc = "-"
        ascii_acc = "-"
        acc_gap = "-"

        if final_adapter.exists():
            eval_cmd = [
                sys.executable, "scripts/evaluate.py",
                "--checkpoint", str(final_adapter),
                "--name", run_id,
                "--test_jsonl", val_split,
                "--output_dir", str(val_eval_file),
                "--num_beams", str(num_beams),
                "--max_new_tokens", str(max_tokens),
            ]
            logger.info(f"Chạy evaluate trên tập Val ({val_split})...")
            eval_ret = subprocess.run(eval_cmd)
            if eval_ret.returncode == 0:
                summary_file = Path(str(val_eval_file).replace(".jsonl", "_summary.json"))
                if summary_file.exists():
                    try:
                        with open(summary_file, "r", encoding="utf-8") as f:
                            s = json.load(f)
                            val_cer = s.get("corpus_cer", "-")
                            val_fffd = s.get("n_replacement_char", "-")
                    except Exception:
                        pass

        # ── PHA 3: Phân tích lỗi & CER Oracle (Chỉ bắt buộc cho lam0) ───
        delta_oracle = "-"
        if r["nesy_weight"] == 0.0 and val_eval_file.exists():
            oracle_dir = run_dir / "oracle"
            oracle_cmd = [
                sys.executable, "analysis/classify_errors_oracle.py",
                "--pred_jsonl", str(val_eval_file),
                "--split_jsonl", val_split,
                "--out_dir", str(oracle_dir),
            ]
            logger.info("Chạy phân loại lỗi và CER Oracle cho checkpoint lam0...")
            oracle_ret = subprocess.run(oracle_cmd)
            if oracle_ret.returncode == 0:
                oracle_summary = oracle_dir / "summary.json"
                if oracle_summary.exists():
                    try:
                        with open(oracle_summary, "r", encoding="utf-8") as f:
                            os_data = json.load(f)
                            delta_oracle = os_data.get("delta_oracle", "-")
                    except Exception:
                        pass

        elapsed = time.time() - start_time
        wall_min = round(elapsed / 60.0, 2)

        # Cập nhật manifest
        row_dict = {
            "run_id": run_id,
            "condition": r["condition"],
            "seed": r["seed"],
            "nesy_weight": r["nesy_weight"],
            "mask_mode": r["mask_mode"],
            "git_commit": _git_commit(),
            "status": "done",
            "wall_min": wall_min,
            "best_step": "-",
            "best_eval_ce": "-",
            "val_corpus_cer": val_cer,
            "val_fffd_lines": val_fffd,
            "byte_acc": byte_acc,
            "ascii_acc": ascii_acc,
            "acc_gap": acc_gap,
            "delta_oracle": delta_oracle,
        }
        _update_manifest(manifest_path, row_dict)

        # Ghi file DONE xác nhận hoàn tất cả 3 pha
        with open(done_file, "w", encoding="utf-8") as f:
            f.write(f"DONE at {time.strftime('%Y-%m-%d %H:%M:%S')} in {elapsed:.1f}s\n")
        logger.info(f"Lượt chạy {run_id} HOÀN TẤT THÀNH CÔNG trong {wall_min} phút.")

    logger.info("TOÀN BỘ CÁC LƯỢT TRONG SWEEP ĐÃ HOÀN TẤT THÀNH CÔNG!")


def main():
    parser = argparse.ArgumentParser(description="Sweep runner cho Bước 3")
    parser.add_argument("--config", type=str, default="configs/sweep_step3.yaml")
    parser.add_argument("--plan", action="store_true", help="In danh sách 15 runs theo thứ tự seed-major")
    parser.add_argument("--status", action="store_true", help="Báo cáo tiến độ các runs")
    parser.add_argument("--dry-run", action="store_true", help="Chạy dry-run thử nghiệm nhẹ trên CPU")
    parser.add_argument("--run", action="store_true", help="Chạy các lượt sweep thực tế")
    parser.add_argument("--only", type=str, default=None, help="Chỉ chạy duy nhất 1 run_id cụ thể")
    parser.add_argument("--force", action="store_true", help="Bỏ qua kiểm tra git clean")
    args = parser.parse_args()

    sweep_cfg = load_sweep_config(args.config)
    runs = build_run_list(sweep_cfg)

    if args.plan or (not args.status and not args.run and not args.dry_run):
        print_plan(runs)
    elif args.status:
        check_status(sweep_cfg, runs)
    elif args.dry_run or args.run:
        run_sweep(sweep_cfg, runs, dry_run=args.dry_run, only=args.only, force=args.force)


if __name__ == "__main__":
    main()
