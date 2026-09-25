# BẢN TỔNG HỢP TIẾP NỐI PHIÊN LÀM VIỆC (SESSION CONTEXT HANDOVER)
*Thời điểm ghi nhận: 16:45 ngày 25/09/2026 (trước khi User chuyển sang Laptop)*

Tài liệu này được tạo tự động nhằm lưu giữ 100% bối cảnh thảo luận, các quyết định học thuật và trạng thái kỹ thuật giữa **User**, **Antigravity (Dev)**, **Claude Code (Architect)** và **Google Jules (Auditor)**.

---

## 1. Bản đồ Trạng thái Dự án (Status Map)

- **Repo**: `LMK89/LV`
- **Nhánh `main`**: Đã chốt và đóng băng **Bước 2b** (commit `d732624`):
  + Đã chia dữ liệu Document-independent (70/15/15 theo mã bài, 0 rò rỉ: Train 936 / Val 202 / Test 212 dòng, 33 cụm test).
  + Đã sửa lỗi DoRA tied-weights `lm_head` (PR #2).
  + Đã hoàn thành bộ công cụ phân loại 5 nhóm lỗi ($a_1, a_2, b_1, b_2, c$) và thuật toán Weighted Levenshtein Oracle CER (PR #3).
  + Đã đóng băng các ngưỡng rẽ nhánh trong `docs/lo_trinh.md`: Trần 2.0% CER, FSM 70%, cận trên 95% CI < 2.0%.
- **Nhánh `exp/04-quet-lambda`**: Đang checkout, đã hoàn tất toàn bộ hạ tầng Bước 3 (commit `aeae488`):
  + Đã có tài liệu thiết kế chi tiết: `docs/designs/step3_lambda_sweep.md`.
  + Đã sửa 2 lỗi chặn thiết kế: Mask random đối chứng công bằng (lấy mẫu trong 49.885 eligible tokens, không phạt EOS/mảnh byte dở dang), loại trừ nhãn đúng khỏi penalty NeSy, log `eval_ce` làm tiêu chí chọn model.
  + Đã có ma trận 15 runs `configs/sweep_step3.yaml`, `configs/train_dryrun.yaml`, và trình điều phối 3 pha `scripts/sweep.py`.
  + Đã hỗ trợ mô hình `F-tiny` ngẫu nhiên cho `--dry-run` không cần tải weights 0.9GB.
  + Toàn bộ 46/46 unit tests trên CPU pass 100%.

---

## 2. Các tài liệu quan trọng đã được tạo trong phiên:
1. `HANDOVER.md`: Hợp đồng bàn giao và phân vai (Claude tư vấn kiến trúc cấp cao, Antigravity thực thi 100%, bảo vệ quota Claude).
2. `docs/reports/email_gui_gvhd.md`: Thư ngỏ gửi Giảng viên Hướng dẫn tiềm năng để xin hướng dẫn và góp ý chuyên môn.
3. `docs/reports/tong_hop_ky_thuat_du_an.md`: Tài liệu kỹ thuật tự đóng gói (Single Source of Truth) cho AI Agent và kỹ sư.
4. `docs/prompts/prompt_for_jules.md`: Prompt kiểm định toàn diện cho Google Jules.
5. `docs/prompts/prompt_for_claude_reports.md`: Prompt hướng dẫn soạn thảo báo cáo cho Claude Code.

---

## 3. Tương tác với Claude Code và Google Jules:
- **Claude Code**:
  + Đang mở tại session: `session_01Fp1aZ4ZebCeY6tWhp7bjPR` trên Edge (Port 9222).
  + Đã viết xong 2 tài liệu báo cáo, giải thích cặn kẽ vì sao Bước 4 (FSM) là dự kiến và chỉ được thực hiện sau khi Bước 3 có số liệu.
- **Google Jules**:
  + Đang audit nhánh `exp/04-quet-lambda`. Đã xác nhận `src/vocr/nesy/loss.py` đạt chuẩn 100%, các script plan/status và oracle hoạt động tốt.
  + Antigravity đã vá `F-tiny` uninitialized model vào `src/vocr/models/loader.py` và `scripts/sweep.py` để Jules và laptop chạy `--dry-run` không bị OOM.

---

## 4. Hướng dẫn hành động trên Laptop mới:
Khi mở máy Laptop, anh chỉ cần:
1. Kéo mã nguồn mới nhất:
   ```bash
   git clone https://github.com/LMK89/LV.git
   cd LV
   git checkout exp/04-quet-lambda
   git pull origin exp/04-quet-lambda
   ```
2. Cài đặt môi trường:
   ```bash
   pip install -r requirements.txt
   ```
3. Chạy test xác nhận môi trường CPU:
   ```bash
   python scripts/run_tests.py
   python scripts/sweep.py --plan
   python analysis/classify_errors_oracle.py --dry-run
   ```
4. Nếu chat với Antigravity trên Laptop:
   Chỉ cần đưa câu lệnh:
   *"Đọc file docs/handovers/session_context_2026_09_25.md và HANDOVER.md để tiếp tục công việc của dự án."*
   Agent mới sẽ lập tức nắm trọn vẹn 100% bối cảnh mà không cần hỏi lại.
