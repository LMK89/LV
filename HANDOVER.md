# HANDOVER — Trạng thái và Hợp đồng bàn giao dự án LV

Dự án: **Ràng buộc cấu trúc âm tiết tiếng Việt cho OCR bằng VLM dùng byte-level BPE: phân tích giới hạn và giải pháp**  
Repo: `LMK89/LV`  
Cập nhật lúc: 25/09/2026 (sau Bước 1b)

---

## 1. Trạng thái hiện tại của các nhánh & công việc

| Bước | Nhánh / PR | Trạng thái | Kết quả chính |
|---|---|---|---|
| **1a** | `exp/01-chia-du-lieu` (PR #1) | **Đã MERGE vào `main`** | Chia 1350 dòng theo mã bài (Document-level grouping): Train 936 (69.3%) / Val 202 (15.0%) / Test 212 (15.7%). Rò rỉ = 0 bài, 0 tài liệu, 0 câu trùng. |
| **1b** | `exp/02-debug-fffd` (PR #2) | **Đã MERGE vào `main`** | **Phát hiện bước ngoặt:** Lỗi U+FFFD (26.7%) ở v1 là do DoRA gắn vào `lm_head` (tied với `decoder.embed_tokens`). Khi gọi `merge_and_unload()`, embedding đầu vào của decoder bị méo. Đã loại bỏ `lm_head` khỏi `configs/dora.yaml`. |
| **2** | `exp/03-phan-loai-loi` (PR #3) | **Đã MERGE vào `main`** | Xây dựng hoàn chỉnh module `src/vocr/eval/oracle.py`, xử lý lỗi tách/gộp từ, Clustered Bootstrap, bộ kiểm thử 24 test cases pass 100%. Sửa lỗi `could_start_valid_syllable` và NeSy loss mask. |
| **2b** | `main` | **BƯỚC TIẾP THEO** | **Chốt ngưỡng đăng ký trước (pre-registration)** trong `docs/lo_trinh.md` TRƯỚC khi có số liệu Bước 3. |
| **3** | `exp/04-quet-lambda` | Sắp tới | Quét $\lambda \in \{0, 0.1, 0.5\} \times \text{mask } \{\text{rule}, \text{random}\} \times 3 \text{ seeds}$ trên GPU. |

---

## 2. Quy tắc vận hành & Ràng buộc hệ thống (Bắt buộc tuân thủ)

1. **Phân công trách nhiệm:**
   - **Claude Code (Opus):** Kiến trúc sư trưởng / Reviewer / Phản biện / Thiết kế tài liệu học thuật.
   - **Antigravity:** Developer / Coder thực thi trực tiếp trên local (code, chạy test, commit và merge).
2. **Nguyên tắc đồng thuận:**
   - Không bên nào tự ý chốt một chiều; mọi thay đổi lớn đều qua thảo luận kỹ thuật, đạt đồng thuận 2 bên mới tiến hành code và merge.
3. **Môi trường máy Antigravity (Local):**
   - **KHÔNG CÓ GPU:** Bỏ qua hoàn toàn các bước train nặng hay nạp full weights trên local.
   - Chỉ chạy các bài kiểm thử nhẹ bằng CPU (unit test, kiểm tra cú pháp, dry-run mock).
4. **Quyết định kỹ thuật đã chốt:**
   - **Bộ chia dữ liệu:** Giữ 1 Fixed Split chuẩn mực 70/15/15 (bác bỏ K-fold).
   - **Đánh giá thống kê:** Dùng Paired Clustered Bootstrap (resample theo Document ID, 33 cụm ở tập test) trong `src/vocr/eval/significance.py`.
   - **Thuật ngữ luận văn:** Công bố là **"Document-independent evaluation"**.
   - **DoRA:** Bỏ `lm_head` khỏi `configs/dora.yaml`, thêm safeguard từ chối merge nếu tied-weights.
   - **Syllables & NeSy:** `could_start_valid_syllable` cho phép `\ufffd` ở cuối chuỗi; NeSy loss mask loại trừ token mảnh byte đơn lẻ.

---

## 3. Nhiệm vụ tiếp theo: BƯỚC 2b (`main`)

- Điền các ngưỡng rẽ nhánh trong bảng quy tắc `docs/lo_trinh.md`:
  - Ngưỡng trần `Δ_oracle` (gợi ý: 2 điểm % CER).
  - Ngưỡng hiệu quả FSM (gợi ý: 70% của `Δ_oracle`).
- Commit chốt ngưỡng trên `main` trước khi tiến hành huấn luyện Bước 3 trên GPU.
