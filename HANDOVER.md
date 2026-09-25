# HANDOVER — Trạng thái và Hợp đồng bàn giao dự án LV

Dự án: **Ràng buộc cấu trúc âm tiết tiếng Việt cho OCR bằng VLM dùng byte-level BPE: phân tích giới hạn và giải pháp**  
Repo: `LMK89/LV`  
Cập nhật lúc: 25/09/2026 (sau Bước 2b, chuẩn bị Bước 3)

---

## 1. Trạng thái hiện tại của các nhánh & công việc

| Bước | Nhánh / PR | Trạng thái | Kết quả chính |
|---|---|---|---|
| **1a** | `exp/01-chia-du-lieu` (PR #1) | **Đã MERGE vào `main`** | Chia 1350 dòng theo mã bài (Document-level grouping): Train 936 (69.3%) / Val 202 (15.0%) / Test 212 (15.7%). Rò rỉ = 0 bài, 0 tài liệu, 0 câu trùng. |
| **1b** | `exp/02-debug-fffd` (PR #2) | **Đã MERGE vào `main`** | **Phát hiện bước ngoặt:** Lỗi U+FFFD (26.7%) ở v1 là do DoRA gắn vào `lm_head` (tied với `decoder.embed_tokens`). Khi gọi `merge_and_unload()`, embedding đầu vào của decoder bị méo. Đã loại bỏ `lm_head` khỏi `configs/dora.yaml`. |
| **2** | `exp/03-phan-loai-loi` (PR #3) | **Đã MERGE vào `main`** | Xây dựng hoàn chỉnh module `src/vocr/eval/oracle.py`, xử lý lỗi tách/gộp từ, Clustered Bootstrap, bộ kiểm thử 24 test cases pass 100%. Sửa lỗi `could_start_valid_syllable` và NeSy loss mask. Đưa tokenizer offline vào `data/tokenizers/`. |
| **2b** | `exp/03b-chot-nguong` | **Đã MERGE vào `main`** (commit `18488b4`) | **Chốt ngưỡng đăng ký trước (pre-registration)** trong `docs/lo_trinh.md` ngày 25/09/2026: Trần $\Delta_{oracle} \ge 2.0$ điểm % CER ($a_1 + seg\_a_1$ trên Val), FSM lấy $\ge 70\%$ của $\Delta_{oracle}$, điều kiện trần thấp khi cận trên 95% CI $< 2.0\%$. Tách hàm `compute_invalid_token_ids` và `build_invalid_mask` trên CPU, 44/44 unit test pass 100%. |
| **3** | `exp/04-quet-lambda` | **ĐANG THỰC HIỆN** | Chuẩn bị sweep $\lambda \in \{0, 0.1, 0.5\} \times \text{mask } \{\text{rule}, \text{random}\} \times 3 \text{ seeds}$ (15 runs). Triển khai dry-run CPU, cấu hình, xử lý đối chứng công bằng cho mask random và tiêu chí chọn checkpoint bằng `eval_ce`. |

---

## 2. Quy tắc vận hành & Phân công vai trò (Cập nhật ngày 25/09/2026)

### 2.1. Phân định vai trò & Tiết kiệm Quota Claude Code
1. **Claude Code:**
   - Từ giờ **thuần túy đóng vai trò Kiến trúc sư tư vấn thiết kế học thuật & giải pháp cấp cao**.
   - **Tuyệt đối ngưng yêu cầu review code dài, log kiểm thử, kết quả chạy test hay nhờ Claude review từng dòng code** để tránh hao tốn quota vô ích.
   - Tập trung vào: phản biện phương pháp luận, thiết kế kiến trúc thí nghiệm, giao thức thống kê, quy tắc rẽ nhánh.
2. **Antigravity:**
   - **Tự chủ 100% khâu hiện thực hóa mã nguồn, viết unit test, chạy kiểm thử, thiết lập kịch bản dry-run nhẹ trên CPU và sửa lỗi kỹ thuật.**
   - Giữ vững tư duy phản biện độc lập với Claude Code; không làm theo một cách thụ động, chỉ triển khai khi đạt đồng thuận kỹ thuật giữa hai bên.
   - Luôn merge code vào nhánh tính năng, tích hợp vào `main` và kéo về local đầy đủ cho User.
3. **Lưu vết thiết kế:**
   - Toàn bộ thiết kế, đề xuất giải pháp và tài liệu kiến trúc từ Claude Code bắt buộc phải được lưu lại đầy đủ tại thư mục `docs/designs/` (ví dụ `docs/designs/step3_lambda_sweep.md`) để User có thể đối chiếu lại giữa thiết kế và code thực tế sau này khi quota hồi phục.

### 2.2. Ràng buộc môi trường & Kỹ thuật
1. **Môi trường máy local Antigravity:**
   - **KHÔNG CÓ GPU:** Tuyệt đối bỏ qua các lệnh huấn luyện nặng hoặc load full weights mô hình lớn trên local.
   - Mọi kiểm thử trên local đều chạy bằng CPU: unit test, script dry-run mock/tiny model. Các lượt huấn luyện 15 runs đầy đủ sẽ chạy trên hạ tầng GPU riêng (Colab / máy chủ).
2. **Quyết định kỹ thuật đã chốt và đóng băng:**
   - **Bộ chia dữ liệu:** Giữ 1 Fixed Split chuẩn mực 70/15/15 theo mã bài (`data/splits/split_report.md`).
   - **Đánh giá thống kê:** Dùng Paired Clustered Bootstrap (resample theo Document ID, 33 cụm ở tập test).
   - **Thuật ngữ luận văn:** Dùng **"Document-independent evaluation"**.
   - **DoRA:** Bỏ hoàn toàn `lm_head` khỏi `configs/dora.yaml`, chặn merge nếu tied-weights.
   - **Ngưỡng rẽ nhánh Bước 2b:** Trần $\Delta_{oracle} \ge 2.0$ điểm % CER ($a_1 + seg\_a_1$ trên Val), FSM $\ge 70\%$, điều kiện trần thấp khi cận trên 95% CI $< 2.0\%$.

---

## 3. Kế hoạch chi tiết Bước 3 (`exp/04-quet-lambda`)

1. **Khắc phục 2 lỗi chặn thiết kế trước khi chạy GPU:**
   - **B1 (Mask random công bằng):**
     + Chỉ lấy mẫu ngẫu nhiên trong 49.885 token hợp lệ (loại trừ token đặc biệt như EOS/BOS và mảnh byte dở dang mang `\ufffd`).
     + Luôn tính vị trí tha thứ ngữ cảnh bằng mask rule dùng chung cho cả 2 điều kiện (rule và random).
     + Không bao giờ phạt token đúng của nhãn ($s_t = \sum_{v \in INVALID \setminus \{y_t\}} p_t(v)$) để tránh phạt oan tên riêng/viết tắt.
   - **B2 (Tiêu chuẩn chọn checkpoint):** Dùng `eval_ce` (chỉ Cross-Entropy loss) làm tiêu chí early stopping và chọn checkpoint tốt nhất giữa các $\lambda$ để đảm bảo thước đo đồng nhất.
2. **Mã nguồn và Cấu hình Bước 3:**
   - `configs/sweep_step3.yaml`: Danh sách 15 runs theo thứ tự seed-major.
   - `configs/train_dryrun.yaml`: Cấu hình chạy dry-run 1 step trên CPU.
   - `scripts/sweep.py`: Trình điều phối chạy sweep (hỗ trợ `--plan`, `--dry-run`, `--run`, `--status`).
   - Cơ chế Dry-run F-tiny trên CPU không cần tải full weights.
