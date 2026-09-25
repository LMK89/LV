# Báo cáo chia dữ liệu

- Nhãn: `data/raw/labels.jsonl` (1350 dòng)
- Khóa nhóm: `article` · tỉ lệ mục tiêu 0.7,0.15,0.15
- Seed chọn: 165 (tốt nhất trong 0..199, điểm 0.296)

| Tập | Số dòng | Tỉ lệ | Số bài | Số tài liệu | Bài lớn nhất |
|---|---|---|---|---|---|
| train | 936 | 69.3% | 31 | 182 | BCCTC (11%) |
| val | 202 | 15.0% | 6 | 31 | 9539 (28%) |
| test | 212 | 15.7% | 7 | 33 | 25849 (28%) |

## Kiểm tra rò rỉ (phải bằng 0)

| Cặp | Bài chung | Tài liệu chung | Câu trùng nguyên văn |
|---|---|---|---|
| train ∩ val | 0 | 0 | 0 |
| train ∩ test | 0 | 0 | 0 |
| val ∩ test | 0 | 0 | 0 |
