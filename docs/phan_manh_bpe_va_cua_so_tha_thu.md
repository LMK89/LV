# Phân mảnh token BPE của tiếng Việt và lựa chọn cửa sổ tha thứ

> Tài liệu phương pháp cho luận văn. Giải thích: (1) vì sao phải đo **số token
> mà BPE cắt mỗi âm tiết**, (2) **phương pháp và cách chạy** để đo, (3) **lý do
> chọn cửa sổ tha thứ = 5 token**, và (4) phần code phải sửa + **đánh giá ảnh
> hưởng**.

---

## 1. Vì sao phải đo "số token / âm tiết"?

Ràng buộc âm vị học của đề tài được định nghĩa ở **cấp âm tiết** (tập 7.184 âm
tiết hợp lệ). Nhưng Florence‑2 sinh văn bản ở **cấp token BPE**. Hai cấp này
KHÔNG trùng nhau: tokenizer byte‑level BPE của Florence‑2 (họ BART/GPT‑2, gốc
tiếng Anh) cắt một âm tiết tiếng Việt thành **nhiều token con**, vì:

- Ký tự có dấu tiếng Việt (ư, ờ, ệ, đ, ạ…) là **nhiều byte UTF‑8**;
- BPE gốc tiếng Anh **không có luật gộp (merge)** cho các chuỗi byte đó → chúng
  bị tách rời thành từng byte‑token.

Hệ quả: khi áp luật cấp âm tiết xuống cấp token, một **mảnh subword hợp pháp**
(vd `ngh` trong `nghiêng`) tự nó không phải âm tiết hợp lệ nên bị **phạt oan**.
Đây là **sai số xấp xỉ** (approximation error) của NeSy Loss. Muốn định lượng và
giảm sai số này, trước hết phải biết **mỗi âm tiết bị cắt thành mấy token** và
**phân bố** của con số đó — đó là lý do phải đo "số token".

---

## 2. Phương pháp đo

**Tokenizer.** Dùng đúng tokenizer của Florence-2 base — checkpoint
`florence-community/Florence-2-base` (bản native trong transformers, cùng
tokenizer với `microsoft/Florence-2-base`; BART
byte‑level BPE, vocab ~50k). Khi không tải được Florence (offline/máy bị chặn
mạng), dùng **GPT‑2/BART BPE cùng họ** làm proxy — hành vi cắt vụn tiếng Việt
đại diện tốt; số cuối cùng để trích luận văn nên lấy từ chính Florence.

**Tập kiểm.** Chạy trên **toàn bộ 7.184 âm tiết hợp lệ** trong
`src/core/nesy/vietnamese_syllables.txt`. Ưu điểm: (a) **độc lập với dataset**
(không cần ảnh/nhãn), nên tái lập được ở bất kỳ máy nào; (b) phủ **toàn bộ không
gian âm tiết** tiếng Việt, không thiên lệch theo tần suất của một tập text cụ
thể.

**Quy trình.** Với mỗi âm tiết `s`:
1. Thêm khoảng trắng đầu (`" " + s`) để mô phỏng âm tiết đứng **đầu một từ**
   trong câu — byte‑level BPE đánh dấu đầu từ bằng ký hiệu `Ġ`.
2. Token hóa, đếm số token.
3. Gom theo độ dài: 1, 2, 3, … token.

**Phân biệt hai đại lượng.**
- *Số token/âm tiết* (tài liệu này): đặc trưng **cố hữu** của tokenizer, đo trên
  danh sách âm tiết — không phụ thuộc dữ liệu.
- *FPR (False Positive Rate)* của NeSy Loss (`analyze_approximation_error.py`):
  đo trên **text huấn luyện thật**, phụ thuộc tần suất âm tiết + cơ chế tha thứ.
  Số token/âm tiết là **đầu vào** để hiểu và chỉnh FPR.

---

## 3. Cách chạy để xác định số token

### 3.1. Script chuyên dụng (khuyến nghị — độc lập dataset)

```bash
# (a) Dùng ĐÚNG tokenizer Florence-2 (cần transformers + tải được model):
python src/core/nesy/verify_bpe_fragmentation.py

# (b) OFFLINE — đưa vocab.json + merges.txt của một byte-level BPE cùng họ:
python src/core/nesy/verify_bpe_fragmentation.py --vocab enc.json --merges vocab.bpe
```

Đầu ra: vài ví dụ token hóa + **bảng phân bố độ dài token** + tỉ lệ âm tiết bị
cắt ≥2 / ≥4 token.

### 3.2. Phân tích FPR trên dữ liệu thật (kèm cơ chế tha thứ)

```bash
python run_pipeline.py --step analyze_error
# -> outputs/results/nesy_approximation_error_final.md
```

---

## 4. Kết quả đo & lý do chọn cửa sổ = 5 token

### 4.1. Phân bố độ dài token/âm tiết

Đo trên 7.184 âm tiết bằng BPE họ GPT‑2/BART *(số minh họa — chạy lại với
Florence để lấy số chuẩn)*:

| Số token/âm tiết | Số âm tiết | Tỉ lệ |
|---|---|---|
| 1 | 255 | 3.5% |
| 2 | 826 | 11.5% |
| 3 | 1.405 | 19.6% |
| 4 | 1.276 | 17.8% |
| **5** | **2.511** | **35.0%** ← đỉnh (mode) |
| 6 | 677 | 9.4% |
| 7 | 218 | 3.0% |
| 8 | 16 | 0.2% |

Trung bình **4,11 token/âm tiết**, tối đa **8**. **96,5%** âm tiết bị cắt ≥2
token; **65,4%** bị cắt ≥4 token. Chỉ 3,5% (âm tiết ASCII trơn: `a, am, ba,
be`…) là 1 token.

### 4.2. Độ phủ theo cửa sổ W

Cơ chế "tha thứ" soi các **dải W token liên tiếp**: nếu dải decode ra toàn âm
tiết hợp lệ thì tha (bỏ phạt). Một âm tiết được tha **trọn vẹn** khi độ dài token
của nó ≤ W. Độ phủ theo W:

| Cửa sổ W | Âm tiết phủ trọn (≤W token) | Bỏ sót |
|---|---|---|
| 2 | 15,0% | 6.103 |
| 3 | 34,6% | 4.698 |
| 4 | 52,4% | 3.422 |
| **5** | **87,3%** | **911** |
| 6 | 96,7% | 234 |
| 7 | 99,8% | 16 |
| 8 | 100,0% | 0 |

### 4.3. Lý do chọn W = 5 (điểm khuỷu / elbow)

1. **Trùng đỉnh phân bố.** Nhiều âm tiết nhất bị cắt thành đúng **5 token**
   (35%). Cửa sổ 5 ôm trọn nhóm đông nhất này.
2. **Bước nhảy độ phủ lớn nhất ở W=5.** Tăng W từ 4→5 làm độ phủ nhảy
   **52,4% → 87,3% (+34,9 điểm)** — mức tăng biên lớn nhất. Sau đó giảm dần:
   W5→6 chỉ +9,4; W6→7 +3,0; W7→8 +0,2 (quy luật *diminishing returns*).
3. **Cân bằng chi phí.** Chi phí duyệt ~ tuyến tính theo (W−1); W=5 đủ để ôm
   phần lớn mà chưa tốn kém (và đã cache kết quả decode nên rẻ trên thực tế).
4. **Cấu hình được.** W là siêu tham số `forgive_window` (mặc định 5). Nếu đo
   trên Florence thấy đuôi dài hơn, có thể nâng lên **6** để phủ ~97% mà chi phí
   tăng không đáng kể.

> **Lưu ý trung thực.** W=5 vẫn **bỏ sót ~13%** âm tiết dài hơn 5 token (và các
> dải bắc qua ranh giới từ). Phần này là **giới hạn đo được** của phép xấp xỉ,
> phải trình bày như *limitation*, không phải "đã giải quyết trọn vẹn". Số chuẩn
> để trích luận văn lấy từ tokenizer Florence (mục 3.1a).

---

## 5. Phần code phải sửa (B1) — hiện trạng & đánh giá ảnh hưởng

### 5.1. Trước đây (code cũ)

- Cửa sổ tha thứ **cố định = 2** (chỉ soi cặp token liền kề), viết **lặp lại** ở
  3 nơi (`loss.py`, `train_stage2.py`, `analyze_approximation_error.py`).
- Chỉ cứu được âm tiết tách **2 token** → FPR đo được **19,21%** (bản lưu:
  `outputs/results/nesy_approximation_error_win2_synthetic.md`, cửa sổ 2, dữ
  liệu synthetic).

### 5.2. Đã sửa (hiện tại)

- Tách một hàm **dùng chung** `context_forgiven_rows(..., max_window=5)` trong
  `src/core/nesy/rules.py`; `loss.py` và `train_stage2.py` cùng gọi hàm này →
  ba nơi không thể lệch nhau.
- `analyze_approximation_error.py` nới cửa sổ tương ứng (helper `_forgiven_flags`)
  để **FPR đo lại khớp** với thứ mô hình thực sự bị phạt khi train.
- Cửa sổ khai báo bằng tham số `forgive_window` (mặc định 5).

### 5.3. Đánh giá ảnh hưởng

| Tiêu chí | Ảnh hưởng |
|---|---|
| **Đổi cấu trúc?** | **KHÔNG.** Không đụng tokenizer, embedding, định dạng dữ liệu, hay tương thích checkpoint. |
| **FPR** | **Giảm** so với 19,21% (âm tiết 3–5 token nay được cứu). Phải chạy lại `analyze_error` để lấy số mới. |
| **Cần train lại?** | Có — vì loss đổi: **Stage 1 + Stage 2**. Biến thể eval bị dịch: **2, 3, 4, 5**. Biến thể 1, 6 (base thuần, không train) **không đổi**. |
| **File đã sửa** | `rules.py`, `loss.py`, `train_stage2.py`, `analyze_approximation_error.py`. |
| **Rủi ro** | Cửa sổ lớn hơn → thêm lần decode (đã cache, rẻ). Không làm "lọt rác" vì chỉ tha khi dải decode ra **toàn âm tiết hợp lệ**. |

---

## 6. Ghi vào luận văn thế nào

1. **Trình bày như một ablation của chính cơ chế tha thứ**: báo cáo FPR ở
   **W=2 vs W=5** (và có thể W=6) → cho thấy tác động của siêu tham số cửa sổ,
   biến điểm yếu thành phân tích có kiểm soát.
2. **Dùng số của tokenizer Florence**: chạy `verify_bpe_fragmentation.py`
   (mục 3.1a) để lấy bảng 4.1/4.2 chuẩn; số GPT‑2 ở đây chỉ là minh họa cùng họ.
3. **Nêu rõ điều kiện đo**: tập 7.184 âm tiết (độc lập dataset) cho phân bố token;
   FPR đo trên tập train thật.
4. **Trích dẫn tái lập**: chỉ rõ script + tham số `forgive_window`, để hội đồng
   chạy lại được — đúng chuẩn khoa học (đối chiếu `docs/Benchmark.md` §5.9).
5. **Giữ tính trung thực**: khẳng định W=5 phủ ~87% và **còn ~13% bỏ sót** (âm
   tiết ≥6 token / bắc qua ranh giới từ) như limitation đã đo được.

---

### Phụ lục — nguồn số liệu

- Phân bố token (mục 4): `src/core/nesy/verify_bpe_fragmentation.py` trên
  `vietnamese_syllables.txt` (7.184 âm tiết), BPE họ GPT‑2/BART.
- FPR cửa sổ‑2 (archive): `outputs/results/nesy_approximation_error_win2_synthetic.md`.
- FPR production (sẽ tạo): `outputs/results/nesy_approximation_error_final.md`
  (chạy `python run_pipeline.py --step analyze_error`).
