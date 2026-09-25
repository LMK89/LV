# LV — Ràng buộc âm tiết tiếng Việt cho OCR trên VLM dùng byte-level BPE

Mã nguồn luận văn thạc sĩ. Đề tài dự kiến:

> **Ràng buộc cấu trúc âm tiết tiếng Việt cho OCR bằng mô hình thị giác–ngôn ngữ
> dùng byte-level BPE: phân tích giới hạn và giải pháp**

Khởi tạo từ repo đồ án môn học
[`LMK89/CS2225.K2025.Florence2-Vietnamese`](https://github.com/LMK89/CS2225.K2025.Florence2-Vietnamese)
tại commit `9806ccc` (tag `v1-phuong-an-A`). Repo cũ được giữ nguyên làm bản lưu
của phương án A (NeSy Loss + Compound DPO). Những gì đã rút ra từ v1:
[`docs/lich_su_v1.md`](docs/lich_su_v1.md).

## Câu hỏi nghiên cứu

1. **Chẩn đoán** — byte-level BPE của Florence-2 cắt vụn tiếng Việt đến mức nào;
   mặt nạ âm tiết cấp token sai bao nhiêu (tách lỗi do luật và lỗi do cửa sổ W).
2. **Cơ chế** — ràng buộc lúc huấn luyện (teacher forcing) dư thừa hay xung đột
   với cross-entropy; trần lợi ích tối đa của ràng buộc âm tiết (CER oracle).
3. **Giải pháp** — áp ràng buộc ở đúng chỗ có ích (FSM lúc decode, và nếu số
   liệu cho phép: preference optimization trên chuỗi tự sinh).

Lộ trình và quy tắc chốt nhánh: [`docs/lo_trinh.md`](docs/lo_trinh.md).

## Cấu trúc

```
configs/            dora.yaml, train.yaml, augment.yaml
data/raw/           images/ + labels.jsonl (1.350 dòng, trường split_v1 = cách chia cũ)
data/splits/        train/val/test sau khi chia lại theo tài liệu (bước 1)
src/vocr/
  nesy/             syllables.py (từ điển + kiểm tra), constraints.py (LogitsProcessor),
                    loss.py (NeSyTrainer, mask rule/random)
  models/           loader.py (Florence-2 native + DoRA), collator.py
  data/             augment.py
  eval/             metrics.py (NFC, CER macro + corpus), significance.py, compare.py
scripts/            train.py, evaluate.py, run_vietocr_baseline.py
analysis/           verify_bpe_fragmentation.py, analyze_approximation_error.py
docs/               lộ trình, lịch sử v1, ghi chú phân mảnh BPE
tests/              pytest (không cần GPU/model)
```

## Cài đặt

```bash
pip install -r requirements.txt
pip install -e .
pytest
```

Chạy offline: đặt `FLORENCE2_BASE_ID=/đường/dẫn/Florence-2-base` tải sẵn.

## Lệnh chính

```bash
# Augment tập train
python -m vocr.data.augment --config configs/augment.yaml

# Huấn luyện: λ=0 (DoRA thuần), λ>0 mặt nạ luật, λ>0 mặt nạ ngẫu nhiên (control)
python scripts/train.py --nesy_weight 0   --seed 1 --output_dir outputs/checkpoints/lam0_s1
python scripts/train.py --nesy_weight 0.1 --seed 1 --output_dir outputs/checkpoints/lam0.1_s1
python scripts/train.py --nesy_weight 0.1 --mask_mode random --seed 1 --output_dir outputs/checkpoints/rand0.1_s1

# Đánh giá một cấu hình (không có --checkpoint = Florence-2 gốc)
python scripts/evaluate.py --name lam0_s1 --checkpoint outputs/checkpoints/lam0_s1/final_adapter
python scripts/evaluate.py --name lam0_s1+fsm --checkpoint outputs/checkpoints/lam0_s1/final_adapter --constraint automaton

# Kiểm định paired bootstrap
python -m vocr.eval.significance --file_a <eval_A.jsonl> --file_b <eval_B.jsonl>
```

## Quy ước

- Mỗi bước lộ trình làm trên một nhánh `exp/NN-ten-buoc`, merge vào `main` khi xong.
- Tập test chỉ dùng cho số liệu báo cáo; chọn checkpoint và siêu tham số bằng val.
- Mọi lần train ghi `run_info.json` (tham số, commit, config) cạnh checkpoint.
- Adapter (`*.safetensors`) không commit vào git — lưu ở kho ngoài, ghi đường dẫn
  trong `docs/`.
