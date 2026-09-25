# Lộ trình và quy tắc chốt nhánh

## Các bước

| Bước | Nhánh | Việc | Tài nguyên |
|---|---|---|---|
| 1a | `exp/01-chia-du-lieu` | Chia lại train/val/test theo **mã bài** (hết rò rỉ) — `data/splits/split_report.md` | CPU |
| 1b | `exp/02-debug-fffd` | Tìm nguyên nhân output chứa U+FFFD ở v1: decode trên ảnh train, round-trip tokenizer, BOS/decoder start | CPU/GPU nhỏ |
| 2 | `exp/03-phan-loai-loi` | Phân loại lỗi (a) âm tiết không hợp lệ / (b) hợp lệ nhưng sai / (c) chèn-xóa; CER oracle; phân mảnh chéo tokenizer | CPU |
| 2b | `main` | **Chốt ngưỡng bên dưới và commit TRƯỚC bước 3** | — |
| 3 | `exp/04-quet-lambda` | λ ∈ {0, 0.1, 0.5} × mask {rule, random} × 3 seed; bootstrap | GPU |
| 3b | `exp/03-phan-loai-loi` | CER oracle chính thức trên checkpoint λ=0 | CPU |
| 4 | `exp/05-fsm` | FSM chính xác cấp byte (không cho qua U+FFFD vô điều kiện, có lối thoát tên riêng) | GPU (suy luận) |
| 5 | — | Gặp GVHD với số liệu, chốt nhánh theo quy tắc | — |
| 6 | `exp/06-dpo` | Chỉ khi quy tắc cho phép | GPU |

## Ghi chú bước 1a

- Cùng một bài báo được nhiều tài liệu chép lại (48 câu trùng nguyên văn giữa các
  tài liệu, luôn cùng mã bài) → chia theo tài liệu vẫn rò rỉ văn bản; phải chia
  theo mã bài.
- Kết quả: train 936 / val 202 / test 212 dòng; test chỉ có 7 bài, 33 tài liệu.
  Tập nhỏ → hiệu ứng nhỏ (~1–2 điểm CER) khó đạt ý nghĩa thống kê. Nên tìm bản
  đầy đủ của bộ dữ liệu gốc để mở rộng test, hoặc báo cáo hiệu ứng nhỏ nhất phát
  hiện được (MDE) từ phương sai bootstrap.

## Ghi chú bước 1b

Chi tiết: `docs/debug_fffd.md`. Tokenizer không có lỗi (round-trip 0/936 hỏng).
Nghi phạm chính: DoRA gắn vào `lm_head` dùng chung trọng số với embedding decoder,
rồi `merge_and_unload()` lúc đánh giá làm đổi embedding đầu vào (PEFT đã cảnh báo
trong log train v1). **Cần chạy `scripts/debug_fffd.py model` trên GPU để xác
nhận trước bước 3**; `configs/dora.yaml` và `loader.py` của LV vẫn mang lỗi này.

## Quy tắc chốt nhánh (chốt chính thức tại bước 2b ngày 25/09/2026, đóng băng trước bước 3)

Ký hiệu: `Δ_oracle` = CER(λ=0) − CER khi sửa đúng toàn bộ lỗi nhóm (a).
Chỉ số `Δ_oracle` được tính dựa trên phân loại lỗi âm tiết chính ($a_1 + seg\_a_1$) trên tập Val.
Khoảng tin cậy 95% được tính bằng Clustered Bootstrap theo Document ID (paired bootstrap).

| Điều kiện | Nhánh |
|---|---|
| NeSy (rule) tốt hơn λ=0 **và** tốt hơn mask random, p < 0,05 (Holm) | Giữ NeSy trong phần giải pháp |
| Cận trên khoảng tin cậy 95% của `Δ_oracle` < **2.0** điểm % CER | Trần thấp → dồn vào phân tích + so chéo tokenizer + mở rộng vocab |
| `Δ_oracle` ≥ **2.0** điểm % CER và FSM lấy được ≥ **70** % của `Δ_oracle` | Đóng góp chính = FSM + phân tích, bỏ DPO |
| `Δ_oracle` ≥ **2.0** điểm % CER và FSM lấy được < **70** % | Làm DPO trên chuỗi tự sinh (tiêu chí chính CER), bắt buộc đối chứng DPO-chỉ-CER |

Quy tắc đã đạt đồng thuận kỹ thuật giữa Antigravity và Claude Code (Opus 5.5).

Ngày chốt ngưỡng: 25/09/2026 (commit: bước 2b)

