# Lịch sử v1 (đồ án môn học CS2225) — những gì mang sang và rút ra

Nguồn: `LMK89/CS2225.K2025.Florence2-Vietnamese@9806ccc` (tag `v1-phuong-an-A`).
Log thô, checkpoint và file kết quả v1 chỉ nằm ở repo cũ.

## Kết quả v1 (KHÔNG dùng cho luận văn)

Test N=200, trung bình CER theo mẫu:

| Cấu hình | CER | Ghi chú |
|---|---|---|
| Florence-2 gốc (zero-shot) | 100,8 % | |
| Stage 1 (DoRA + NeSy, λ=0,1) | 59,6 % | 199/200 output chứa U+FFFD |
| Stage 1 + ràng buộc decode + retry hint | 1091 % | vòng retry đưa prompt ngoài phân phối |
| Gốc + ràng buộc decode ("Oracle") | 119,3 % | tên "Oracle" sai nghĩa |
| VietOCR (zero-shot) | 31,5 % | |

Lý do không dùng được:

1. **Rò rỉ dữ liệu**: 128/132 tài liệu của test có mặt trong train; 15 dòng test
   trùng nhãn nguyên văn với train.
2. **Thiếu đối chứng λ=0**: "Stage 1 tốt hơn gốc" chỉ chứng minh fine-tune tốt
   hơn zero-shot, không nói gì về NeSy.
3. **Output hỏng** (U+FFFD) dù eval_loss = 0,60 → nghi lệch train/suy luận.

## Chẩn đoán cần sửa lại

- BUG-007 của v1 cho rằng mảnh byte `�` bị đánh dấu invalid và bị phạt. **Sai với
  code**: `is_valid_syllable("�")` trả về True (regex coi là ký tự đặc biệt).
  Token bị phạt thật sự là mảnh đầu như `' ng'` cùng subword tiếng Anh; 45.792 /
  51.290 token (89 %) bị đánh dấu.
- NeSyPenalty chỉ 0,01–0,03 so với CE 0,5–9 → tác động lên gradient rất nhỏ.
- `train_loss` trung bình báo 8,2 trong khi CE cuối ≈ 0,5: cần kiểm tra việc
  `compute_loss` override có chia đúng cho gradient accumulation hay không.

## Thay đổi khi chuyển sang LV

| v1 | LV |
|---|---|
| `core/nesy/rules.py` | `vocr/nesy/syllables.py` + `constraints.py`; bỏ bảng âm vị không dùng, bỏ `VietSyllableValidator`/retry hint; thiếu từ điển thì báo lỗi thay vì rơi về ~50 âm tiết |
| `train_stage1.py` | `scripts/train.py`: `--seed`, `--mask_mode rule/random`, ghi `run_info.json` |
| `evaluate.py` (6 biến thể cố định) | một cấu hình/lần chạy, CER corpus + macro + median, cảnh báo output bất thường |
| `metrics.py` | chuẩn hóa NFC, thêm `corpus_cer` / `corpus_wer` |
| `run_vietocr_baseline.py` | bỏ chế độ "simulation" (sinh dự đoán giả từ nhãn) |
| Stage 2 DPO, demo, quantize, run_pipeline/run_dryrun | không mang sang (xem repo cũ) |

## Bài học kỹ thuật (giữ lại)

- Luôn `truncation=True` khi dựng input; bật `CUDA_LAUNCH_BLOCKING=1` khi debug
  CUDA assert — traceback bất đồng bộ chỉ sai chỗ.
- lm_head của Florence-2 có 51.328 hàng, tokenizer 51.290 token: che 38 id "ma"
  ở mọi bước decode.
- Từ điển phải có cả hai kiểu đặt dấu thanh (hoà/hòa); bóc dấu câu trước khi tra.
