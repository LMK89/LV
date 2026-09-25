# Đặc tả Nhiệm vụ Bước 2: Phân loại lỗi và Tính trần CER Oracle

Nhánh mục tiêu: `exp/03-phan-loai-loi`  
Tài nguyên: **100% CPU** (không cần GPU)

---

## 1. Bối cảnh & Mục tiêu khoa học

Theo lộ trình tại `docs/lo_trinh.md`, trước khi tiến hành quét $\lambda$ tốn kém ở Bước 3, luận văn cần trả lời câu hỏi cơ chế cốt tử:
> **"Nếu một bộ giải mã hoàn hảo (Oracle) sửa được toàn bộ các âm tiết tiếng Việt vi phạm cấu trúc ngữ âm/từ điển, thì CER tối đa có thể giảm được bao nhiêu điểm %?"**

Khoảng giảm này chính là:
$$\Delta_{\text{oracle}} = \text{CER}_{\text{base}} - \text{CER}_{\text{oracle}}$$

Đây là **trần lý thuyết (theoretical ceiling)** cho mọi phương pháp ràng buộc cấu trúc (dù dùng NeSy Loss lúc train hay FSM lúc decode). Nếu $\Delta_{\text{oracle}}$ quá bé (< 1-2 điểm %), việc ép ràng buộc âm tiết sẽ không mang lại nhiều ý nghĩa thực tế. Nếu $\Delta_{\text{oracle}}$ đủ lớn, đây là bảo chứng khoa học vững chắc cho giải pháp FSM / NeSy.

---

## 2. Các yêu cầu kỹ thuật cụ thể

### 2.1. Phân loại lỗi theo 3 nhóm (Error Taxonomy)
Xây dựng module/script phân loại chi tiết từng ký tự/âm tiết sai giữa Ground Truth và Output dự đoán (sử dụng Levenshtein alignment cấp ký tự/âm tiết):
1. **Nhóm (a) - Âm tiết không hợp lệ (Invalid Syllable):**
   - Chuỗi ký tự không tuân thủ quy tắc ngữ âm tiếng Việt hoặc không có trong từ điển âm tiết chuẩn (`src/vocr/nesy/vietnamese_syllables.txt`).
   - Bao gồm cả các ký tự vỡ mã UTF-8 (`U+FFFD`).
   - Đây là đối tượng duy nhất mà FSM / Ràng buộc âm tiết có thể can thiệp sửa chữa.
2. **Nhóm (b) - Âm tiết hợp lệ nhưng sai ngữ cảnh (Valid but Wrong Syllable):**
   - Viết đúng âm tiết tiếng Việt nhưng sai so với nhãn (ví dụ: nhãn "hoa", model nhận diện thành "hóa" hoặc "họa").
   - Nhóm này FSM bất lực vì bản thân âm tiết hoàn toàn hợp lệ.
3. **Nhóm (c) - Lỗi chèn / xóa (Insertion / Deletion):**
   - Mất từ hoặc sinh thừa ký tự rác.

### 2.2. Thuật toán đo CER Oracle
- Giả lập bộ sửa lỗi hoàn hảo cho Nhóm (a):
  - Với mọi âm tiết trong dự đoán thuộc Nhóm (a), thay thế bằng âm tiết Ground Truth tương ứng tại vị trí đó (nếu có alignment tương ứng).
  - Tính lại CER sau khi đã sửa toàn bộ lỗi Nhóm (a).
- Tính $\Delta_{\text{oracle}} = \text{CER}_{\text{raw}} - \text{CER}_{\text{oracle}}$.
- Báo cáo số liệu trên tập test 212 dòng (hoặc tập val 202 dòng).

### 2.3. Phân mảnh chéo Tokenizer (Cross-Tokenizer Fragmentation)
- So sánh mức độ phân mảnh của byte-level BPE (Florence-2) với các tokenizer khác (ví dụ: SentencePiece/Unigram của VietOCR hoặc PhoBERT/mBART).
- Đo số lượng token trung bình trên một âm tiết tiếng Việt có dấu.

---

## 3. Sản phẩm bàn giao yêu cầu từ Claude Code

1. Tạo nhánh `exp/03-phan-loai-loi` từ `main`.
2. Tạo script phân tích: `analysis/classify_errors_oracle.py` (chạy hoàn toàn bằng CPU, có cờ `--dry-run` hoặc chạy trên file kết quả predictions có sẵn).
3. Viết tài liệu thiết kế và báo cáo: `docs/designs/step2_error_analysis.md`.
4. Viết unit test kiểm tra thuật toán phân loại và tính CER Oracle trong `tests/test_oracle.py`.
5. Mở Pull Request vào `main` để Antigravity review.
