# PROMPT GỬI CLAUDE CODE: SOẠN THẢO 2 TÀI LIỆU HỌC THUẬT & KỸ THUẬT DỰ ÁN

```markdown
Chào Claude (Kiến trúc sư trưởng),
Với vai trò là Kiến trúc sư học thuật của dự án Luận văn Thạc sĩ:
"Ràng buộc cấu trúc âm tiết tiếng Việt cho OCR bằng VLM dùng byte-level BPE: phân tích giới hạn và giải pháp" (Repo: LMK89/LV),
nhờ bạn soạn thảo chính thức 2 tài liệu markdown sau đây và lưu vào thư mục `docs/reports/`:

================================================================================
TÀI LIỆU 1: `docs/reports/email_gui_gvhd.md`
(Thư ngỏ gửi Giảng viên Hướng dẫn tiềm năng để xin hướng dẫn và góp ý chuyên môn)
================================================================================
Yêu cầu:
1. Ngữ điệu: Chuẩn mực học thuật, khiêm tốn, trang trọng, cô đọng (khoảng 500-700 từ), cấu trúc rõ ràng để Thầy/Cô có thể nắm bắt nhanh.
2. Nội dung cần truyền tải:
   - Giới thiệu bản thân và lý do liên hệ: Mong muốn xin phép được Thầy/Cô nhận làm Giảng viên Hướng dẫn cho Luận văn Thạc sĩ.
   - Tính cấp thiết & Điểm nghẽn học thuật:
     + Các Vision-Language Models (VLM) hiện đại (tiêu biểu như Florence-2) sử dụng tokenizer byte-level BPE, khiến các ký tự tiếng Việt có dấu bị phân rã thành nhiều mảnh byte riêng lẻ (ví dụ "nghiêng" bị cắt thành 3-4 token).
     + Hậu quả: Mô hình dễ sinh ra chuỗi byte lỗi (U+FFFD), sinh từ vi phạm cấu trúc ngữ âm học tiếng Việt (sai phụ âm đầu, vần không tồn tại, dấu thanh đặt sai vị trí).
   - Giải pháp và Đóng góp đề xuất:
     + Tinh chỉnh hiệu quả bằng DoRA (đã giải mã và xử lý triệt để lỗi tied-weights giữa `lm_head` và decoder embedding từ phiên bản v1).
     + NeSy Loss (Neuro-Symbolic Constraint): Cơ chế phạt khả vi trong quá trình train đối với các token tạo âm tiết sai, có cửa sổ tha thứ ngữ cảnh subword (Context-forgiveness window W=5) và nhóm đối chứng ngẫu nhiên công bằng (Fair Random Control).
     + FSM Constrained Decoding: Cưỡng chế cấu trúc âm tiết tiếng Việt cấp byte lúc suy luận.
     + Phương pháp đánh giá mới: Weighted Levenshtein DP bóc tách 5 nhóm lỗi âm tiết và đo CER Oracle (giới hạn trần cải thiện lý thuyết).
   - Giao thức thực nghiệm chuẩn mực:
     + Đánh giá độc lập tài liệu (Document-independent split: chia theo mã bài, 0 rò rỉ văn bản).
     + Phân tích ý nghĩa thống kê bằng Paired Clustered Bootstrap theo Document ID trên CER toàn corpus với hiệu chỉnh Holm.
   - Lời ngỏ: Trân trọng xin lịch hẹn gặp / trao đổi online để báo cáo chi tiết và xin ý kiến định hướng từ Thầy/Cô.

================================================================================
TÀI LIỆU 2: `docs/reports/tong_hop_ky_thuat_du_an.md`
(Tài liệu kỹ thuật tổng hợp toàn diện — Single Source of Truth Context Package cho AI Agent & Kỹ sư)
================================================================================
Yêu cầu:
Tài liệu tự đóng gói hoàn chỉnh (Self-contained). Bất kỳ một AI Agent hoặc Chuyên gia kỹ thuật nào chỉ cần đọc duy nhất file này là nắm trọn vẹn 100% bản chất dự án, kiến trúc, phương pháp luận, số liệu thực nghiệm và các tham số cấu hình mà KHÔNG CẦN tốn token đọc toàn bộ repo.

Cấu trúc bắt buộc gồm 6 phần:
1. Tổng quan đề tài & Kiến trúc nền tảng:
   - Bài toán: OCR tài liệu tiếng Việt bằng Vision-Language Model.
   - Mô hình nền tảng: Florence-2-base (0.23B params, DaViT vision encoder + BART-like decoder).
   - Tokenizer: Byte-level BPE, vocab size = 51.290 tokens. Cơ chế phân mảnh nguyên âm có dấu.
   - Phân tích nguyên nhân lỗi U+FFFD ở v1: DoRA gắn vào `lm_head` có `tie_word_embeddings=True`, khi gọi `merge_and_unload()` làm méo embedding decoder. Cách xử lý ở v2: loại bỏ `lm_head` khỏi `configs/dora.yaml`, tiết kiệm ~1.30M params, suy luận an toàn 100%.

2. Dữ liệu & Giao thức chia tập (Document-Independent Split):
   - Tổng cộng 1.350 dòng OCR thực tế.
   - Chia 70/15/15 theo **Mã bài viết (Document-level grouping)**: Train 936 dòng (69.3%) / Val 202 dòng (15.0%) / Test 212 dòng (15.7%).
   - Hoàn toàn độc lập: 0 bài chung, 0 tài liệu chung, 0 câu trùng lặp giữa các tập (loại trừ hoàn toàn rò rỉ văn bản do cùng bài báo được nhiều người chép lại). Tập test gồm 33 cụm tài liệu.

3. Phương pháp NeSy Loss (Neuro-Symbolic Constraint):
   - Công thức hàm mất mát tổng quát: $\mathcal{L} = \mathcal{L}_{CE} + \lambda \frac{\sum_t m_t \cdot s_t}{\sum_t m_t + \epsilon}$.
   - Công thức penalty không phạt nhãn đúng: $s_t = \sum_{v \in INVALID \setminus \{y_t\}} p_t(v)$.
   - Tập `eligible`: 49.885 token hợp lệ (loại bỏ special tokens và mảnh byte dở dang mang U+FFFD).
   - Tập `rule` invalid: 45.775 token vi phạm quy tắc âm tiết tiếng Việt.
   - Nhóm đối chứng `random` công bằng: Lấy mẫu 45.775 token từ `eligible`, không bao giờ phạt EOS (`</s>`), dùng chung vị trí tha thứ ngữ cảnh theo rule, độ tương đồng Jaccard ~0.85 so với rule mask.
   - Cửa sổ tha thứ ngữ cảnh subword (Context-forgiveness window W=5): Không phạt các mảnh subword riêng lẻ nếu chúng ghép lại thành âm tiết hợp lệ.
   - Tiêu chí chọn checkpoint thống nhất: Dùng `eval_ce` (chỉ CE loss thuần) làm `metric_for_best_model` cho tất cả các giá trị $\lambda$.

4. Phương pháp CER Oracle & Phân loại 5 nhóm lỗi âm tiết:
   - Thuật toán: Weighted Levenshtein DP ở cấp âm tiết (xử lý bóc tách lỗi tách từ $1:k$ và gộp từ $k:1$).
   - Taxonomy 5 nhóm:
     + $a_1$: Âm tiết không hợp lệ cấu trúc tiếng Việt (lỗi ngữ âm - mục tiêu chính của NeSy/FSM).
     + $a_2$: Âm tiết ngoài từ điển nhưng cấu trúc hợp lệ (tên riêng, từ mượn ngoại lai).
     + $b_1$: Sai dấu thanh điệu.
     + $b_2$: Sai ký tự chữ cái nhưng vẫn tạo âm tiết hợp lệ.
     + $c$: Lỗi chèn / xóa ký tự hoặc lỗi phân đoạn từ ($seg\_a_1$).
   - Chỉ số trần cải thiện lý thuyết: $\Delta_{\text{oracle}} = \text{CER}(\lambda=0) - \text{CER}(\text{sửa hoàn toàn } a_1 + seg\_a_1)$.

5. Giao thức Đánh giá Thống kê & Ngưỡng Rẽ nhánh Đăng ký trước (Pre-registration):
   - Paired Clustered Bootstrap theo Document ID trên CER toàn corpus (10.000 samples, 33 cụm test).
   - Hiệu chỉnh Holm cho 2 phép so sánh trên Test: Rule($\lambda^*$) vs $\lambda=0$ và Rule($\lambda^*$) vs Random($\lambda^*$).
   - Bảng quy tắc rẽ nhánh (đã đóng băng ngày 25/09/2026):
     + Ngưỡng trần: $\Delta_{\text{oracle}} \ge 2.0$ điểm % CER ($a_1 + seg\_a_1$ trên Val).
     + Điều kiện trần thấp: Cận trên khoảng tin cậy 95% của $\Delta_{\text{oracle}} < 2.0$ điểm % CER $\rightarrow$ Trần thấp, chuyển hướng phân tích vocab/tokenizer.
     + Vùng chưa định nghĩa (H4): $\Delta < 2.0$ nhưng cận trên 95% CI $\ge 2.0 \rightarrow$ Chưa đủ bằng chứng trần thấp, tiếp tục đo hiệu quả FSM.
     + Ngưỡng FSM: FSM lấy được $\ge 70\%$ của $\Delta_{\text{oracle}} \rightarrow$ Đóng góp chính là FSM + phân tích, bỏ DPO.
     + FSM lấy được $< 70\% \rightarrow$ Làm DPO trên chuỗi tự sinh đối chứng DPO-chỉ-CER.

6. Thiết kế Thí nghiệm Bước 3 (`exp/04-quet-lambda`):
   - Ma trận 15 runs: $\lambda \in \{0, 0.1, 0.5\} \times \text{mask} \{\text{rule}, \text{random}\} \times 3 \text{ seeds}$ (order: `seed_major`).
   - Cấu hình: `configs/sweep_step3.yaml`, `configs/train_dryrun.yaml`, `configs/train.yaml`.
   - Trình điều phối: `scripts/sweep.py` với các chế độ `--plan`, `--status`, `--run`, `--dry-run`.
   - 5 cổng kiểm soát GPU (G0 $\rightarrow$ G1 $\rightarrow$ G2 $\rightarrow$ G3 $\rightarrow$ G4 $\rightarrow$ G5).

Sau khi hoàn tất, bạn hãy xác nhận và lưu trực tiếp vào 2 file tương ứng trong `docs/reports/`.
```
