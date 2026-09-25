# Báo cáo Phân loại Lỗi & Trần CER Oracle (Bước 2)

- **Tổng số câu đánh giá**: 10
- **Tổng số ký tự Ground Truth**: 250
- **CER ban đầu (Raw)**: 7.200%
- **CER sau khi sửa Oracle (Nhóm a1)**: 5.200%
- **Trần can thiệp tối đa (Δ_oracle)**: **+2.000 điểm % CER**

---

## 1. Phân bố các nhóm lỗi (Error Taxonomy)

| Nhóm | Ý nghĩa | Số lượng | Tỷ lệ / tổng lỗi | Khả năng can thiệp |
|---|---|---|---|---|
| **a1** | Âm tiết sai không hợp lệ, nhãn hợp lệ | 5 | 50.0% | **FSM / NeSy sửa được** |
| **a2** | Âm tiết sai không hợp lệ, nhãn ngoài từ điển | 1 | 10.0% | Cần cơ chế thoát (Bailout) |
| **b1** | Hợp lệ nhưng sai dấu thanh | 1 | 10.0% | Mô hình ngôn ngữ / LM |
| **b2** | Hợp lệ nhưng sai chữ cái | 1 | 10.0% | Không thể sửa bằng từ điển |
| **c** | Lỗi chèn / xóa ký tự | 2 | 20.0% | Căn chỉnh hình ảnh |

---

## 2. Kết luận định hướng cho Luận văn

- Khoảng trần **Δ_oracle = 2.000%** cho thấy mức cải thiện tối đa về mặt lý thuyết nếu toàn bộ lỗi vi phạm cấu trúc tiếng Việt được khắc phục hoàn toàn.
- Nếu **Δ_oracle ≥ 2.0%**, đây là bảo chứng khoa học vững chắc để phát triển giải pháp FSM decoding và NeSy Loss ở Bước 3 & Bước 4.
