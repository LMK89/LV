# HANDOVER — Trạng thái và Hợp đồng bàn giao dự án LV

Dự án: **Ràng buộc cấu trúc âm tiết tiếng Việt cho OCR bằng VLM dùng byte-level BPE: phân tích giới hạn và giải pháp**  
Repo: `LMK89/LV`  
Cập nhật lúc: 25/09/2026 (sau Bước 1b)

---

## 1. Trạng thái hiện tại của các nhánh & công việc

| Bước | Nhánh / PR | Trạng thái | Kết quả chính |
|---|---|---|---|
| **1a** | `exp/01-chia-du-lieu` (PR #1) | **Đã MERGE vào `main`** | Chia 1350 dòng theo mã bài (Document-level grouping): Train 936 (69.3%) / Val 202 (15.0%) / Test 212 (15.7%). Rò rỉ = 0 bài, 0 tài liệu, 0 câu trùng. |
| **1b** | `exp/02-debug-fffd` (PR #2) | **Đã MERGE vào `main`** | **Phát hiện bước ngoặt:** Lỗi U+FFFD (26.7%) ở v1 là do DoRA gắn vào `lm_head` (tied với `decoder.embed_tokens`). Khi gọi `merge_and_unload()`, embedding đầu vào của decoder bị méo. Model và NeSy Loss không bị lỗi. |
| **2** | `exp/03-phan-loai-loi` | **SẮP BẮT ĐẦU** | Phân loại lỗi 3 nhóm, CER Oracle (trần can thiệp cấu trúc), phân mảnh chéo tokenizer. |

---

## 2. Quy tắc vận hành & Ràng buộc hệ thống (Bắt buộc tuân thủ)

1. **Môi trường máy Antigravity (Local):**
   - **KHÔNG CÓ GPU:** Bỏ qua hoàn toàn các bước train nặng hay nạp full weights trên local.
   - Chỉ chạy các bài kiểm thử nhẹ bằng CPU (unit test `pytest`, kiểm tra cú pháp, dry-run 1-2 mẫu mock).
2. **Kỷ luật Git & Phối hợp:**
   - Mọi bước làm việc đều mở nhánh riêng `exp/NN-ten-buoc`.
   - Claude Code viết tài liệu thiết kế vào `docs/designs/` và script vào `src/` hoặc `scripts/`, sau đó tạo PR.
   - Antigravity sẽ review, merge vào `main` và kéo về local.
3. **Quyết định kiến trúc đã chốt:**
   - **Bộ chia dữ liệu:** Giữ 1 Fixed Split chuẩn mực 70/15/15 (bác bỏ K-fold).
   - **Đánh giá thống kê:** Dùng Paired Clustered Bootstrap (resample theo Document ID, 33 cụm ở tập test) trong `src/vocr/eval/significance.py`.
   - **Thuật ngữ luận văn:** Công bố là **"Document-independent evaluation"**.
   - **Sửa lỗi DoRA lm_head:**
     - Với checkpoint cũ: Đặt `merge=False` khi nạp để đánh giá.
     - Với cấu hình huấn luyện mới: Bỏ `lm_head` khỏi `target_modules` trong `configs/dora.yaml`.

---

## 3. Nhiệm vụ tiếp theo: BƯỚC 2 (`exp/03-phan-loai-loi`)

- **Mục tiêu:** Phân tích dữ liệu & dự đoán v1 (hoặc baseline) để tính trần CER Oracle trước khi bước vào huấn luyện Bước 3.
- **Tài nguyên yêu cầu:** Hoàn toàn chạy trên **CPU** (rất phù hợp cho máy hiện tại).
- **Đặc tả chi tiết:** Xem tại `specs/step2_oracle_cer.md`.
