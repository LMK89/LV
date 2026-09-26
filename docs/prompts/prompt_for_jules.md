# PROMPT DÀNH CHO GOOGLE JULES (AUDIT & VERIFICATION)

Copy toàn bộ nội dung bên dưới và dán vào Jules (hoặc tạo Issue trên GitHub để Jules thực thi):

```markdown
Chào Jules, bạn là Kỹ sư Đảm bảo Chất lượng & Kiểm định Học thuật (Independent Academic Code Auditor).
Hãy tiến hành rà soát toàn diện (Comprehensive Audit) dự án Luận văn Thạc sĩ tại repository này trên nhánh `exp/04-quet-lambda`.

---

### I. THÔNG TIN BỐI CẢNH DỰ ÁN
- **Đề tài**: "Ràng buộc cấu trúc âm tiết tiếng Việt cho OCR bằng VLM dùng byte-level BPE: phân tích giới hạn và giải pháp"
- **Mô hình**: Florence-2 (DaViT vision encoder + BART text decoder, byte-level BPE tokenizer 51.290 tokens).
- **Phương pháp**: DoRA adapter, NeSy Loss (Neuro-symbolic constraint), FSM decoding cấp byte, Weighted Levenshtein Oracle CER.
- **Tài liệu cốt lõi cần đọc trước khi audit**:
  1. `HANDOVER.md`: Hợp đồng bàn giao và các quy tắc hệ thống.
  2. `docs/lo_trinh.md`: Lộ trình và các ngưỡng rẽ nhánh đăng ký trước (pre-registration).
  3. `docs/designs/step3_lambda_sweep.md`: Tài liệu thiết kế chi tiết Bước 3 (Quét λ × mask × seed).
  4. `docs/designs/step2_error_analysis.md`: Thiết kế phân loại 5 nhóm lỗi âm tiết và CER Oracle.

---

### II. NHIỆM VỤ 1: ĐỐI CHIẾU MÃ NGUỒN VỚI THIẾT KẾ (GAP ANALYSIS)
Hãy đối chiếu chi tiết giữa tài liệu thiết kế `docs/designs/step3_lambda_sweep.md` và mã nguồn hiện thực:
1. **Module NeSy Loss (`src/vocr/nesy/loss.py`)**:
   - `compute_eligible_token_ids`: Đã lọc đúng 49.885 token hợp lệ (loại trừ special tokens và mảnh byte dở dang mang U+FFFD) chưa?
   - `compute_invalid_token_ids`: Đã lọc đúng 45.775 token vi phạm quy tắc âm tiết tiếng Việt chưa?
   - `build_invalid_mask`: Mask `random` đối chứng đã được lấy mẫu công bằng từ tập `eligible` (không phạt EOS/special tokens, giữ nguyên `rule_invalid_ids` để tha thứ ngữ cảnh) chưa?
   - `compute_loss`: Đã loại trừ token đúng của nhãn khỏi penalty ($s_t = \sum_{v \in INVALID \setminus \{y_t\}} p_t(v)$) để tránh phạt oan tên riêng/viết tắt chưa?
   - `eval_ce`: Đã được tính và log riêng biệt để làm tiêu chí `metric_for_best_model` chưa?
2. **Cấu hình & Điều phối Sweep**:
   - `configs/sweep_step3.yaml`: Ma trận 15 runs theo thứ tự seed-major (`lam0`, `lam0.1_rule`, `lam0.1_rand`, `lam0.5_rule`, `lam0.5_rand` × 3 seeds) có khớp thiết kế không?
   - `configs/train_dryrun.yaml` và `configs/train.yaml`: Các trường cấu hình (`metric_for_best_model: eval_ce`, `bf16`, v.v.) có đồng bộ không?
   - `scripts/train.py`: Đã hỗ trợ `max_train_samples`, `max_eval_samples`, `max_steps`, kiểm tra `torch.cuda.is_bf16_supported()`, và lưu tỉ lệ trainable parameters chưa?
   - `scripts/sweep.py`: Các chế độ `--plan`, `--status`, `--run`, `--dry-run` có đảm bảo tính idempotent và quản lý checkpoint đúng thiết kế không?

---

### III. NHIỆM VỤ 2: ĐÁNH GIÁ KỸ THUẬT / GIẢI PHÁP / HỌC THUẬT
Hãy phản biện độc lập các quyết định phương pháp luận:
1. **Tính công bằng của nhóm đối chứng (Rule vs. Random Control)**:
   - Việc cố định tập mẫu trong 49.885 token eligible và dùng chung vị trí tha thứ theo luật âm tiết có giải quyết triệt để vấn đề "random là một lambda yếu hơn 6 lần" chưa?
   - Độ tương đồng Jaccard ~0.85 giữa mask rule và random có đủ để phân tách tác dụng của "tri thức âm tiết" với "bộ điều chuẩn chung" không?
2. **Loại trừ nhãn đúng khỏi Penalty**:
   - Dưới teacher forcing, việc loại bỏ xác suất nhãn đúng khỏi penalty NeSy có gây rủi ro toán học hoặc gradient vanishing nào không?
3. **Giao thức Thống kê & Ngưỡng Rẽ nhánh (`docs/lo_trinh.md`)**:
   - Paired Clustered Bootstrap theo Document ID trên CER toàn corpus (33 cụm tập test) có đảm bảo tính độc lập tài liệu (Document-independent) không?
   - Quy tắc rẽ nhánh (ngưỡng trần 2.0% CER, FSM 70%, cận trên 95% CI < 2.0%) có đủ chặt chẽ để bảo vệ luận văn trước hội đồng không?

---

### IV. NHIỆM VỤ 3: DRY-RUN TOÀN BỘ DỰ ÁN
Hãy thực thi các lệnh kiểm thử sau và xác nhận đầu ra:
1. **Chạy toàn bộ Test Suite**:
   ```bash
   python scripts/run_tests.py
   ```
   (Yêu cầu: Xác nhận 46/46 tests across 5 files đều PASS).
2. **Chạy thử kịch bản Kế hoạch Sweep**:
   ```bash
   python scripts/sweep.py --plan
   python scripts/sweep.py --status
   ```
   (Yêu cầu: Kiểm tra danh sách 15 runs không trùng lặp, đúng thứ tự seed-major).
3. **Chạy thử mô phỏng phân loại lỗi Oracle**:
   ```bash
   python analysis/classify_errors_oracle.py --dry-run
   ```
   (Yêu cầu: Xác nhận xuất bảng thống kê 5 nhóm lỗi và bootstrap CER không lỗi encoding).

---

### V. KẾT QUẢ ĐẦU RA YÊU CẦU
Hãy trả về một bản Báo cáo Kiểm định (Audit Report) gồm:
1. **Bảng đánh giá mức độ tuân thủ (Compliance Scorecard)**: Từng tiêu chí ghi rõ [PASS], [WARNING], hoặc [FAIL].
2. **Bất kỳ điểm lệch (Gaps / Discrepancies)**: Giữa thiết kế và code thực tế (nếu phát hiện).
3. **Khuyến nghị cải tiến**: Các điểm cần lưu ý trước khi mang lên cụm máy chủ GPU huấn luyện 15 lượt chạy thực tế.
```
