# Thiết kế Bước 2: Phân loại lỗi, CER Oracle, phân mảnh chéo tokenizer

Nhánh: `exp/03-phan-loai-loi` · Đặc tả gốc: `specs/step2_oracle_cer.md`
Trạng thái: **Đã đồng thuận mục A1–A6, nhãn phụ b1/b2 và ngưỡng báo động C3.2 (25/09/2026).**
Đã có code: `src/vocr/eval/oracle.py`, `analysis/classify_errors_oracle.py`, `tests/test_oracle.py`.
Kết quả dry-run và kiểm chứng ở mục D, cuối tài liệu.
Tài nguyên: 100 % CPU, không nạp weight model.

Tài liệu này có 3 phần: (A) các điểm cần chốt trước khi code, (B) thiết kế chi
tiết, (C) phản hồi về quyết định bỏ `lm_head` khỏi DoRA. Mục A là phần cần hai
bên trả lời. Mục B mô tả cách làm theo phương án tôi đề xuất cho từng điểm.

---

## A. Các điểm cần đồng thuận trước khi code

| # | Câu hỏi | Đề xuất của Claude | Lý do ngắn |
|---|---|---|---|
| A1 | Lấy dự đoán từ đâu để tính Δ_oracle? | Bước 2 chỉ **xây công cụ và kiểm chứng nó**. Số Δ_oracle chính thức lấy ở bước 3b, trên checkpoint λ=0 (đúng như `docs/lo_trinh.md`). | Không có file dự đoán hợp lệ nào chạy được bằng CPU (xem B0). |
| A2 | Δ_oracle chính gồm những lỗi nào? | Chỉ các **phép thay thế** mà âm tiết dự đoán không hợp lệ **và âm tiết nhãn hợp lệ** (nhóm a1). Báo cáo thêm 2 cận trên. | FSM chặt không sinh được âm tiết ngoài từ điển, nên không sửa được lỗi mà nhãn là tên riêng. |
| A3 | Căn chỉnh âm tiết theo Levenshtein chuẩn hay có trọng số? | Có trọng số: chi phí thay thế = khoảng cách ký tự chuẩn hóa. Có thêm một lượt nhận diện lỗi tách/gộp từ. | Căn chuẩn ghép cặp tùy tiện khi có nhiều nghiệm tối ưu. Oracle ngây thơ có thể làm CER **tăng** (B3.3). |
| A4 | Sửa hàm `is_valid_syllable` cho U+FFFD thế nào? | Công cụ bước 2 tự kiểm U+FFFD trước. Việc sửa `syllables.py` tách thành một commit riêng cần hai bên duyệt, vì nó đổi hành vi của NeSy và FSM. | Hàm hiện coi `"�"`, `"��"`, `"ú�"` là **hợp lệ** (đã chạy, xem B2.2). |
| A5 | `allow_domain_tokens` bật hay tắt? | Dùng **cùng giá trị** với FSM và NeSy ở bước 3 (hiện là `True`). Chạy thêm một lần với `False` để đo độ nhạy. | Trần phải đo bằng đúng định nghĩa "hợp lệ" mà phương pháp can thiệp sẽ dùng. |
| A6 | Đo Δ_oracle trên val hay test? | Dùng **val** cho quyết định rẽ nhánh (bước 2b→3b). Test chỉ để báo cáo cuối. | Δ_oracle quyết định hướng đi tiếp theo, nên không được nhìn vào test. |

---

## B. Thiết kế chi tiết

### B0. Nguồn dự đoán: hiện chưa có file nào dùng được cho số chính thức

| Nguồn | Chạy bằng CPU? | Dùng được cho Δ_oracle? |
|---|---|---|
| `eval_2. Stage1_NeSyLoss_…jsonl` của v1 | Có (file đã có sẵn) | **Không.** Output bị hỏng do lỗi merge (26,7 % ký tự là U+FFFD), làm nhóm (a) phình to giả tạo. Tập chia v1 cũng bị rò rỉ và không khớp tập test mới. Chỉ dùng làm **dữ liệu kiểm tra sức chịu tải** cho công cụ. |
| Florence-2 zero-shot trên val/test mới | Cần nạp weight (HANDOVER cấm làm trên máy local) | Không đại diện. Phân bố lỗi của model chưa fine-tune khác hẳn model đã fine-tune. |
| v1 nạp với `merge=False` trên tập mới | Cần GPU | Không. Train của v1 có thể trùng với val/test mới. |
| **Checkpoint λ=0 của bước 3** | Chỉ khâu suy luận cần GPU. Phân tích chạy bằng CPU. | **Có.** Đây là số chính thức (bước 3b). |

Hệ quả: đầu ra của bước 2 gồm (1) công cụ đã có unit test, (2) báo cáo phân
mảnh tokenizer (chỉ cần file tokenizer, không cần weight), (3) chạy thử trên
output v1 để kiểm chứng. Ngưỡng ở bước 2b là **đăng ký trước** nên không cần
biết Δ_oracle. Như vậy thứ tự của lộ trình vẫn giữ nguyên.

Nếu muốn có số ước lượng sớm, cần **một** lượt suy luận GPU của λ=0 trên val.
Lượt này tự có khi chạy bước 3.

### B1. Đầu vào và đầu ra

```
python analysis/classify_errors_oracle.py \
    --pred_jsonl outputs/results/eval_<name>_<ts>.jsonl \   # định dạng của scripts/evaluate.py
    --split_jsonl data/splits/val.jsonl \                   # ghép image -> document để bootstrap theo cụm
    --out_dir outputs/analysis/step2/<name> \
    [--no-domain-tokens] [--dry-run]
```

- Đọc hai khóa `reference` và `predicted` (đúng định dạng `evaluate.py` ghi ra).
  Ghép với `document` qua `image`, vì file dự đoán không có mã tài liệu.
- `--dry-run`: chạy trên khoảng 10 cặp mẫu viết sẵn trong code, không đọc file
  nào. Dùng để kiểm tra nhanh trên máy local.
- Đầu ra:
  - `lines.jsonl`: với mỗi dòng, danh sách các cặp đã căn chỉnh `(ref, hyp, op, nhóm)`,
    số edit trước và sau oracle.
  - `summary.json`: toàn bộ số liệu ở B4.
  - `report.md`: bảng để dán thẳng vào luận văn, kèm 20 ví dụ cho mỗi nhóm.

### B2. Đơn vị và chuẩn hóa

#### B2.1. Âm tiết

Âm tiết là một khối `\S+`, **giữ lại vị trí ký tự `(start, end)`** trong
chuỗi gốc. Khi sửa ở B3, ta thay đúng đoạn ký tự đó. Phần còn lại của chuỗi
giữ nguyên từng byte. Cách này tránh việc `" ".join()` âm thầm chuẩn hóa khoảng
trắng rồi làm CER đổi vì lý do không liên quan đến việc sửa.

Mọi chuỗi đều được đưa về NFC (`metrics.normalize_text`). CER vẫn phân biệt
hoa/thường như trong `metrics.py`. Riêng việc kiểm tra hợp lệ thì không phân
biệt hoa/thường (theo `is_valid_syllable`).

#### B2.2. Kiểm tra hợp lệ và lỗ hổng U+FFFD (đã xác nhận)

```
is_valid_syllable('�')  -> True     is_valid_syllable('chng�') -> False
is_valid_syllable('��') -> True     is_valid_syllable('l�a')   -> False
is_valid_syllable('ú�') -> True     is_valid_syllable('(�')    -> True
```

Nguyên nhân: U+FFFD thuộc lớp `\W`. Do đó regex "chỉ gồm dấu câu" khớp với nó,
và bước bóc dấu câu ở đầu/cuối cũng cắt mất nó.

Trong công cụ bước 2, hàm phân loại dùng
`is_invalid(tok) = ("�" in tok) or not is_valid_syllable(tok, allow_domain_tokens=…)`.

Lỗ hổng này **cũng có trong NeSy Loss và FSM**, vì cả hai cùng gọi
`is_valid_syllable`. Nó khớp với ghi chú "không cho qua U+FFFD vô điều kiện" ở
bước 4 của lộ trình. Đề xuất sửa gốc trong `syllables.py` bằng một commit riêng
có test. Commit đó cần hai bên đồng ý trước bước 3 (xem A4), vì nó làm thay đổi
tập token bị NeSy phạt.

### B3. Căn chỉnh và phân loại

#### B3.1. Căn chỉnh ở cấp âm tiết

Dùng quy hoạch động Levenshtein trên hai dãy âm tiết, với chi phí:

- Chèn hoặc xóa: 1.
- Thay thế: `0` nếu hai âm tiết bằng nhau, ngược lại là
  `cer(r, h) = edit_distance(r, h) / max(len(r), len(h))`, nằm trong khoảng (0, 1].

Có trọng số thì "cái gì thay cho cái gì" rõ ràng. Ví dụ, `a b c` → `x c` được
ghép `a→x` hay `b→x` tùy chữ nào gần hơn, thay vì tùy thứ tự duyệt. Khi hòa
điểm, ưu tiên thay thế, rồi xóa, rồi chèn, để kết quả luôn tất định.

Độ phức tạp là O(n·m) với n, m ≈ 15 âm tiết mỗi dòng. Chạy thuần Python là đủ,
không cần thêm thư viện (`rapidfuzz` chưa có trong `requirements.txt`).

#### B3.2. Nhãn cho từng cặp

| Phép | Điều kiện | Nhóm |
|---|---|---|
| khớp | `r == h` | (không lỗi) |
| thay thế | `is_invalid(h)` và `not is_invalid(r)` | **a1**: FSM chặt sửa được về nguyên tắc |
| thay thế | `is_invalid(h)` và `is_invalid(r)` (tên riêng, từ nước ngoài, ký hiệu) | **a2**: chỉ sửa được nếu có lối thoát tên riêng |
| thay thế | `not is_invalid(h)` | **b**: hợp lệ nhưng sai |
| xóa | — | **c-del** |
| chèn | `is_invalid(h)` (rác, U+FFFD) | **c-ins-invalid** |
| chèn | `not is_invalid(h)` | **c-ins-valid** |
| tách/gộp | xem B3.3 | **seg** (ghi riêng, tính là nhóm a nếu phía dự đoán không hợp lệ) |

Nhóm (b) có thêm hai nhãn phụ **chỉ để thống kê**: `b-case` (chỉ khác
hoa/thường) và `b-punct` (chỉ khác dấu câu dính kèm, ví dụ `linh.` → `linh`).
Cả hai đều nằm ngoài khả năng của FSM âm tiết. Tách ra để khỏi lẫn với lỗi
"sai dấu thanh" (`hoa`→`hóa`), vốn là phần lõi của nhóm (b).

Nếu cần mịn hơn nữa, có thể tách nhóm (b) theo **khác biệt chỉ nằm ở dấu**
(bỏ dấu hai bên thì bằng nhau) hay **khác cả chữ cái**. Nhãn này hữu ích cho
luận văn nhưng không bắt buộc (xem câu hỏi mở cuối tài liệu).

Spec xếp "U+FFFD" vào nhóm (a) và "ký tự rác sinh thừa" vào nhóm (c), nên hai
định nghĩa chồng nhau. Bảng trên giải quyết chỗ đó: thế chỗ một âm tiết thì là
(a), sinh thêm không có âm tiết nhãn tương ứng thì là `c-ins-invalid`.

#### B3.3. Lỗi tách/gộp và vì sao oracle ngây thơ có thể làm CER tăng

Ví dụ đã chạy bằng `metrics.edit_distance`:

| Nhãn | Dự đoán | Edit gốc | Oracle ngây thơ (`chúngta`→`chúng`, `ta` bị xóa) | Oracle nhận biết gộp từ |
|---|---|---|---|---|
| `chúng ta ăn` | `chúngta ăn` | **1** | **3** (tăng) | **0** |

Căn chỉnh âm tiết biến một lỗi gộp từ (thiếu đúng 1 dấu cách) thành "thay thế
+ xóa". Thay `chúngta` bằng `chúng` sẽ lộ ra phép xóa `ta` và làm CER **tăng**.
Nhưng `chúngta` lại là âm tiết không hợp lệ, tức đúng loại lỗi mà FSM sẽ chặn.

Thiết kế xử lý:

1. **Lượt nhận diện tách/gộp** chạy sau DP. Ghép k:1 hoặc 1:k với k ≤ 3 nếu
   `"".join(ref[i:i+k]) == hyp[j]` (gộp) hoặc `ref[i] == "".join(hyp[j:j+k])`
   (tách). Nếu thỏa, nhóm đó thành một đơn vị `seg`. Oracle sẽ thay bằng
   `" ".join(ref[i:i+k])`.
2. **Chốt chặn theo dòng:** `edits_oracle(dòng) = min(edits_gốc, edits_sau_sửa)`,
   đồng thời đếm số dòng rơi vào trường hợp "sửa xong còn tệ hơn".
   Chọn không can thiệp nằm trong quyền của một oracle, nên lấy min là hợp lệ.
   Nếu số dòng này khác 0 trên dữ liệu thật thì phải đọc lại từng dòng. Đây là
   dấu hiệu căn chỉnh còn sai.

### B4. CER Oracle và các con số báo cáo

Mọi CER đều là **CER cấp corpus (micro)**, dùng `metrics.corpus_cer`, cùng hàm
với số chính của `evaluate.py`.

| Ký hiệu | Định nghĩa | Vai trò |
|---|---|---|
| `CER_raw` | CER của dự đoán gốc | mốc so sánh |
| `CER_oracle` | Sửa **a1 + seg có phía dự đoán không hợp lệ** | **Δ_oracle chính** (đưa vào quy tắc rẽ nhánh) |
| `CER_oracle_a` | Sửa thêm a2 | cận trên nếu có lối thoát tên riêng hoàn hảo |
| `CER_oracle_a+ins` | Sửa thêm: xóa các `c-ins-invalid` | cận trên lỏng nhất. FSM chặn được âm tiết rác nhưng khó đoán nó sẽ sinh gì thay vào |

Δ = `CER_raw − CER_oracle*`, tính bằng điểm %.

**Khoảng tin cậy:** bootstrap theo cụm **tài liệu**, chọn lại các tài liệu có
hoàn lại, 10.000 lần, lấy khoảng phân vị 95 %. Mỗi lần tính lại cả hai CER cấp
corpus trên tập tài liệu đã chọn.

Lưu ý: HANDOVER ghi `significance.py` đã làm "Paired Clustered Bootstrap theo
Document ID". Nhưng bản trên `main` (dòng 74) đang lấy mẫu **theo từng dòng**
(`np.random.choice(n, …)`) và không đọc `document`. Tôi sẽ viết hàm bootstrap
theo cụm dùng riêng cho bước 2. Việc sửa `significance.py` cần hai bên xác nhận
riêng: có thể bản theo cụm nằm ở nhánh khác chưa merge.

**Rủi ro ngược mà FSM mang theo** (oracle chỉ đo phần lợi, nên phải báo cáo
cùng):

- Tỉ lệ âm tiết **nhãn** không hợp lệ (tên riêng, từ nước ngoài) trên val/test.
  Đây là phần văn bản mà FSM không có lối thoát sẽ phá hỏng.
- Số âm tiết dự đoán **đã đúng** nhưng không hợp lệ (khớp nhãn và nhãn nằm ngoài
  từ điển). FSM chặt sẽ làm hỏng những âm tiết này.

**Phân rã CER theo nhóm** (xấp xỉ): mỗi cặp thay thế đóng góp `edit_distance(r, h)`,
mỗi phép chèn hoặc xóa đóng góp `len + 1` (tính cả dấu cách). Tổng các đóng góp
là cận trên của edit cấp ký tự thật. Phần chênh được ghi ra để người đọc biết
phân rã này là xấp xỉ. Con số Δ_oracle **không** dùng phân rã này. Nó được tính
bằng cách sửa chuỗi rồi đo lại CER.

### B5. Phân mảnh chéo tokenizer (spec 2.3)

Chỉ cần file tokenizer (`tokenizer.json` hoặc `vocab`+`merges`, cỡ vài MB),
**không cần weight**. Làm được trên CPU, không vi phạm HANDOVER. Máy của
Claude hiện không tải được từ huggingface.co. Nếu máy Antigravity có mạng thì
chạy ở đó, hoặc commit sẵn file tokenizer vào `data/tokenizers/` (xem câu hỏi mở).

Mở rộng `analysis/verify_bpe_fragmentation.py`:

| Tokenizer | Loại | Vì sao đưa vào |
|---|---|---|
| Florence-2 (BART) | byte-level BPE, 51k | đối tượng chính |
| Qwen2-VL | byte-level BPE, 151k, đa ngữ | **đối chứng quan trọng:** cùng là byte-BPE nhưng vocab phủ tiếng Việt. Tách được yếu tố "byte-level" khỏi yếu tố "vocab thiếu tiếng Việt" |
| mBART-50 / XLM-R | SentencePiece Unigram, 250k | đa ngữ, không cắt giữa ký tự |
| PhoBERT | BPE trên tiếng Việt đã tách từ, 64k | tokenizer bản ngữ (lưu ý: cần tách từ trước) |
| VietOCR | bảng ký tự | mốc dưới: 1 token = 1 ký tự, không bao giờ vỡ UTF-8 |

Số liệu, đo ở **hai mức**:

1. **Theo loại**: trên 7.184 âm tiết của từ điển (script hiện có đã làm với
   Florence).
2. **Theo tần suất**: trên các âm tiết thật của val và test. Mức này sát với
   OCR hơn, vì âm tiết phổ biến thường ít bị cắt vụn hơn.

Với mỗi mức, báo cáo: số token trung bình trên âm tiết (tách riêng âm tiết
**có dấu** và **không dấu**); phân bố 1, 2, 3 và 4+ token; và **tỉ lệ âm tiết
có ranh giới token nằm giữa một ký tự UTF-8**. Chỉ số cuối là nguồn gốc cơ học
của U+FFFD, và chỉ tokenizer cấp byte mới có nó.

Mỗi âm tiết được mã hóa với một dấu cách đứng trước (giống vị trí giữa câu). Đo
thêm dạng viết hoa chữ đầu, vì đầu dòng OCR thường viết hoa.

### B6. Unit test (`tests/test_oracle.py`)

Mọi test chạy bằng CPU, không cần model:

1. Căn chỉnh: dự đoán trùng nhãn thì Δ = 0 và không có lỗi nào.
2. `hoa`→`hóa` thuộc nhóm **b**. `hoa`→`hoá` (dấu kiểu cũ) cũng là **b**,
   không phải a.
3. `người`→`ngươì` thuộc **a1**, và oracle sửa về 0 edit.
4. `Obama`→`Obamn` thuộc **a2**: không tính vào Δ chính, có trong `CER_oracle_a`.
5. Mọi âm tiết chứa U+FFFD (`"�"`, `"ú�"`, `"(�"`) đều được xếp là không hợp
   lệ (khóa lại lỗ hổng ở B2.2).
6. Trường hợp gộp từ `chúng ta ăn` / `chúngta ăn` thuộc **seg**, oracle về 0.
   Đồng thời có test chứng minh oracle ngây thơ cho 3 edit.
7. Xóa hoặc chèn thuộc **c**. `c-ins-invalid` chỉ bị bỏ ở `CER_oracle_a+ins`.
8. Chốt chặn: `CER_oracle ≤ CER_raw` trên mọi dòng, kiểm bằng dữ liệu sinh ngẫu
   nhiên (thêm, xóa, đổi dấu ngẫu nhiên trên câu thật của val).
9. Sửa theo vị trí ký tự: dự đoán có hai dấu cách liền nhau ở phần không bị sửa
   thì phần đó vẫn giữ nguyên hai dấu cách.
10. Chuẩn hóa: nhãn NFC và dự đoán NFD của cùng một chữ được tính là khớp.
11. Bootstrap theo cụm: nếu mọi tài liệu giống hệt nhau thì khoảng tin cậy suy
    biến về một điểm. Cùng seed thì cho cùng kết quả.

### B7. Kế hoạch commit (sau khi chốt mục A)

1. `analysis/classify_errors_oracle.py` + `tests/test_oracle.py`, gồm các mục B1 đến B4 và B6.
2. Mở rộng `analysis/verify_bpe_fragmentation.py` (B5).
3. Chạy thử trên output v1 (để kiểm tra sức chịu tải) và `--dry-run`, rồi ghi
   kết quả vào `docs/designs/step2_error_analysis.md` (phần "Kết quả").
4. *(Commit riêng, cần hai bên duyệt):* sửa U+FFFD trong `syllables.py`, và nếu
   cần, bootstrap theo cụm trong `significance.py`.
5. Mở PR vào `main`.

---

## C. Phản hồi phản biện: bỏ `lm_head` khỏi `target_modules`

**Kết luận: đồng ý bỏ `lm_head` cho mọi lần train từ bước 3.** Có ba chỉnh sửa
nhỏ về lập luận, cộng với một phép kiểm tra rủi ro cho tiếng Việt. Phép kiểm
tra này không tốn thêm lượt chạy.

### C1. Chỗ đồng ý

- Merge an toàn là yêu cầu thật. `merge_and_unload()` với một module dùng chung
  trọng số chính là nguyên nhân gây U+FFFD ở v1. Bỏ `lm_head` thì loại được lỗi
  này tận gốc, thay vì phải nhớ "không bao giờ merge" ở mọi script.
- Quy ước phổ biến có trọng lượng. Ví dụ, chế độ `target_modules="all-linear"`
  của PEFT **cố ý loại lớp output**. Tôi không kiểm được các bài báo cụ thể từ
  máy này, nên nếu luận văn trích dẫn thì cần nguồn cụ thể. Nên viết là "theo
  quy ước của PEFT/QLoRA", tránh viết "mọi nghiên cứu chuẩn". Nhiều notebook
  fine-tune Florence-2 phổ biến **có** gắn `lm_head`. Danh sách target cũ của v1
  chép từ một notebook như vậy (xem chú thích trong `configs/dora.yaml`).
- Vì kết quả v1 đã bị loại (rò rỉ dữ liệu cộng với lỗi merge), việc đổi cấu hình
  adapter **không làm mất khả năng so sánh** với số liệu nào còn giá trị.

### C2. Ba chỗ nên chỉnh trong lập luận

1. **"Bất đối xứng kiến trúc" không phải lỗi khi không merge.** Khi chưa merge,
   DoRA trên `lm_head` biến model thành model *không dùng chung trọng số*: đầu
   vào dùng bảng embedding gốc, đầu ra dùng `W + ΔW`. Đó là một model hợp lệ và
   nhất quán giữa train và suy luận (bảng 4 của `debug_fffd.py model` dự kiến
   cho thấy unmerged sạch). Lỗi **chỉ** phát sinh khi merge. Lý do đúng để bỏ là
   "để merge an toàn và triển khai đơn giản", không phải "kiến trúc sai".
2. **Trọng số dùng chung không chỉ với `decoder.embed_tokens`.** Trong BART
   (text tower của Florence-2), `shared` là embedding chung của **cả encoder lẫn
   decoder**, và `lm_head` dùng chung với nó. Vì vậy việc merge ở v1 cũng làm
   méo embedding của prompt `<OCR>` bên encoder. Ảnh hưởng nhỏ, vì prompt chỉ có
   vài token. Tên module chính xác ở bản native cần xác nhận bằng mục 1 của
   `debug_fffd.py model`.
3. **"PEFT không hỗ trợ merge an toàn" là quá mạnh.** Cảnh báo trong
   `train_v2.log` nhắc đến tùy chọn `ensure_weight_tying`. PEFT có cơ chế để giữ
   hai phía dùng chung trọng số đồng bộ. Tôi **chưa kiểm** cơ chế này có hỗ trợ
   DoRA trên lớp dùng chung với phiên bản PEFT đang pin (`peft>=0.10`, chưa khóa
   bản) hay không. Dù sao đây cũng là lựa chọn phức tạp hơn, nên không đổi kết
   luận. Chỉ nên sửa câu chữ nếu đưa vào luận văn.

Dung lượng giảm được: `lm_head` cỡ 768 × 51.290. Với r = 24, riêng phần này là
24·(768 + 51.290) + 51.290 (vector độ lớn của DoRA) ≈ **1,30 triệu tham số**.
Tỉ lệ so với toàn adapter cần đo bằng `print_trainable_parameters()`, vì vision
tower cũng được gắn adapter. Tôi chưa có số này.

### C3. Rủi ro cho khả năng học chữ tiếng Việt: có, mức vừa, và đo được

Cơ chế: logit của token t là `h · e_t`, với `e_t` là hàng embedding dùng chung
(bị đóng băng). DoRA trên `lm_head` cung cấp hai thứ:

- **Vector độ lớn theo từng hàng.** Về bản chất đây là "nhiệt độ" hay "độ ưu
  tiên" học được cho từng token. Nó rất tiện để nâng các mảnh byte tiếng Việt
  (UTF-8 byte 0x80–0xFF, như `á`, `»`, `º`…), vốn hiếm trong dữ liệu tiền huấn
  luyện chủ yếu là tiếng Anh. BART có `final_logits_bias`, nhưng đó là buffer,
  không train được.
- Một hạng r xoay hướng các hàng output.

Bỏ `lm_head` thì model phải mã hóa phần ưu tiên đó qua hidden state `h`, thông
qua `fc1/fc2/out_proj` của decoder. Không gian này vẫn đủ 768 chiều sau 6 lớp,
nên **về lý thuyết vẫn học được**. Nhưng có thể chậm hơn, đặc biệt nếu embedding
của các byte hiếm được huấn luyện kém (hiện tượng "under-trained tokens").

Không bỏ `lm_head` cũng không giải quyết được hạn chế lớn hơn: **embedding đầu
vào** của các mảnh byte đó vẫn đóng băng trong cả hai cấu hình, vì adapter chưa
merge không đụng tới phía đầu vào. Vậy trên phía đầu vào, bỏ `lm_head` không
làm tình hình tệ hơn.

Cách đo, không tốn thêm lượt chạy riêng:

1. Trong lượt λ=0 đầu tiên ở bước 3, log thêm **độ chính xác teacher forcing
   tách riêng token mảnh byte và token ASCII** trên val, cùng tỉ lệ U+FFFD và CER
   trên các ký tự có dấu. Công cụ bước 2 sẽ in sẵn tỉ lệ nhóm a1 có dấu và
   không dấu.
2. Ngưỡng báo động (đề xuất, cần chốt): val có U+FFFD > 1 % số dòng, **hoặc**
   độ chính xác của token mảnh byte thấp hơn token ASCII hơn 15 điểm %. Khi đó
   mới xét phương án dự phòng.
3. **Phương án dự phòng giữ được việc dùng chung trọng số:** train riêng các
   **hàng embedding** của mảnh byte tiếng Việt (vài trăm id), bằng
   `trainable_token_indices` của PEFT (TrainableTokens), thay vì gắn DoRA vào
   `lm_head`. Vì hàng embedding là dùng chung, đầu vào và đầu ra được cập nhật
   **nhất quán**, và merge chỉ là ghi đè vài hàng, nên an toàn. Cần kiểm phiên
   bản PEFT có hỗ trợ tính năng này và có tôn trọng việc dùng chung trọng số
   hay không trước khi dùng. Chi phí khoảng 300 × 768 ≈ 0,23 triệu tham số, nhỏ
   hơn 1,3 triệu của DoRA trên `lm_head`.

Đi kèm quyết định này, tôi đề xuất (trong một commit riêng, không nằm trong
bước 2):

- `configs/dora.yaml`: bỏ `lm_head`.
- `loader.load_lora_model`: **từ chối merge** khi `adapter_config.json` có một
  module dùng chung trọng số trong `target_modules` và
  `tie_word_embeddings=True`. Đây là lớp phòng thủ thứ hai cho các checkpoint
  cũ, để tránh lặp lại lỗi âm thầm.
- `task_type` (`CAUSAL_LM` → `SEQ_2_SEQ_LM`) là quyết định **tách riêng**, vì nó
  đổi cách PEFT bọc model. Nếu định đổi thì nên đổi cùng lúc với `lm_head`,
  trước lượt train đầu tiên của bước 3, để mọi cấu hình λ đều giống nhau.

---

## Câu hỏi mở cho Antigravity

1. Mục A1 đến A6: đồng ý hay có phương án khác?
2. Nhóm (b) có cần tách nhãn phụ "chỉ khác dấu" và "khác chữ cái" trong luận
   văn không?
3. Máy local có tải được tokenizer từ Hugging Face không? Nếu không, ta commit
   các file tokenizer (vài MB, không phải weight) vào `data/tokenizers/`.
4. Ai sửa `syllables.py` (U+FFFD) và `significance.py` (bootstrap theo cụm)?
   Sửa ở nhánh nào? Tôi đề xuất một nhánh nhỏ `exp/03b-sua-kiem-tra` merge trước
   bước 3.
5. Chốt các ngưỡng báo động ở C3.2.

---

## D. Hiện thực và kết quả dry-run (sau đồng thuận)

### D1. Các điểm đã chốt và cách hiện thực

| Điểm | Hiện thực |
|---|---|
| A1: bước 2 chỉ làm công cụ | Số Δ_oracle chính thức vẫn chờ checkpoint λ=0 (bước 3b), đo trên **val** (A6). |
| A2: Δ chính = a1 | `ORACLE_MAIN = {a1, seg_a1}`. Hai cận trên: `oracle_a` (thêm a2, seg_a2) và `oracle_a_ins` (xóa thêm `c_ins_invalid`). |
| A3: căn chỉnh | DP Levenshtein trên âm tiết với chi phí nguyên. Chèn/xóa = 1000, thay thế = 1000·khoảng cách ký tự chuẩn hóa. Tách/gộp k:1 và 1:k (k ≤ 3) có chi phí `1 + 1000·d`, chỉ cho phép khi `d ≤ 1/3`, với `d` là khoảng cách giữa chuỗi nối lại và phía bên kia. Khi hòa điểm, ưu tiên thay thế, rồi tách/gộp, rồi xóa, rồi chèn. |
| Chốt chặn | Mỗi mức oracle lấy `min(edit mức trước, edit sau khi sửa)`. Các mức lồng nhau nên luôn có `oracle_a_ins ≤ oracle_a ≤ oracle ≤ raw`. Số dòng mà chốt chặn phải can thiệp được báo cáo riêng (`lines_worse_*`). |
| A4: U+FFFD | `is_invalid()` tự kiểm U+FFFD, ngoài việc gọi `is_valid_syllable` (đã được vá ở 283511a). |
| A5 | Cờ `--no-domain-tokens`, mặc định bật (giống FSM/NeSy). |
| Nhãn phụ nhóm b (đồng thuận mới) | **b1 = sai dấu**: bằng nhau sau khi bỏ dấu thanh và mũ/móc/trăng, với `đ` coi như `d` có dấu. Có tách riêng `detail=tone` (chỉ khác 5 dấu thanh) và `detail=diacritic`. **b2 = sai chữ cái.** Hai nhãn thống kê riêng là `b_case` (chỉ khác hoa/thường) và `b_punct` (chỉ khác dấu câu dính kèm). Nhóm a1 cũng mang `detail` để biết âm tiết không hợp lệ do sai dấu hay do rác/U+FFFD (`other`). |
| Bootstrap | Chọn lại **tài liệu** có hoàn lại, tính lại Δ trên CER cấp corpus, lấy khoảng phân vị 95 % (mặc định 10.000 lần). Mã tài liệu lấy từ `--split_jsonl`; nếu không có thì dùng `significance.extract_cluster_id`. |
| Phân mảnh | Đọc offline `data/tokenizers/tokenizer.json` bằng gói `tokenizers`, không cần torch. Báo cáo tỉ lệ sai và tỉ lệ nhóm a theo số token BPE của âm tiết nhãn (1, 2, 3, 4+). |

Logic nằm ở `src/vocr/eval/oracle.py` để test import được và để bước 3b dùng
lại. File `analysis/classify_errors_oracle.py` chỉ đọc/ghi và in báo cáo. Đầu
ra gồm `summary.json`, `report.md` và `lines.jsonl` (các cặp lỗi của từng dòng
cùng chuỗi sau khi sửa bằng oracle).

### D2. Kết quả dry-run (10 dòng mock, 3 tài liệu)

`python analysis/classify_errors_oracle.py --dry-run`. Đây là mock nên số liệu
**chỉ để kiểm tra công cụ**, không phải kết quả thí nghiệm.

| Mức | CER | Δ (điểm %) | Dòng mà sửa làm tệ hơn |
|---|---|---|---|
| raw | 10,00 | — | — |
| oracle (a1 + seg_a1) | 7,00 | 3,00 | 0 |
| oracle_a | 6,50 | 3,50 | 0 |
| oracle_a_ins | 4,00 | 6,00 | 0 |

Mỗi dòng mock rơi đúng vào nhóm đã định: `ngươì` → a1, `hóa` → b1,
`chúngta` → seg_a1, `Obamn` → a2, `chng�`/`l�a` → a1 (other), `xyzq` →
c_ins_invalid, `học` bị mất → c_del, `bát` → b2, `Năm/năm` → b_case,
`tuổi,/tuổi` → b_punct. Tổng edit phân rã theo nhóm (20) khớp với edit thật (20).

### D3. Kiểm chứng trên nhiễu tổng hợp (202 câu val × 5 seed = 1.010 dòng)

Dùng bộ sinh nhiễu của `tests/test_oracle.py`: đổi dấu, xóa hoặc chèn ký tự,
gộp hoặc tách từ, chèn U+FFFD.

- Thời gian chạy khoảng 13 giây cho 1.010 dòng (thuần Python, CPU).
- Chốt chặn phải can thiệp ở **2/1.010 dòng**. Cả hai là kiểu gộp từ "gần
  đúng" vượt ngưỡng `d ≤ 1/3`. Ví dụ `dịp viếng` → `dịpqviế gơ`: căn thành
  `dịp→dịpqviế` (a1) và `viếng→gơ` (b), nên sửa a1 riêng lẻ làm lộ ra phần
  thiếu. Chốt chặn giữ nguyên các dòng này, nên Δ không bị thổi phồng. Khi chạy
  trên dự đoán thật, nếu `lines_worse_oracle` lớn hơn khoảng 1 % số dòng thì
  cần đọc lại `lines.jsonl`.
- Δ trên nhiễu tổng hợp không mang ý nghĩa khoa học, vì tỉ lệ các loại lỗi do
  bộ sinh nhiễu quyết định.

### D4. Số liệu thật không phụ thuộc dự đoán (nhãn val/test + tokenizer Florence-2)

| | val | test |
|---|---|---|
| Số âm tiết nhãn | 3.167 | 3.321 |
| Âm tiết nhãn ngoài từ điển, `allow_domain_tokens` bật / tắt | **1,11 % / 1,14 %** | **0,96 % / 1,11 %** |
| Ví dụ ngoài từ điển | NVƠNN, VN, Kanni, H., m2 | Cienco, UBND, TP, TP.HCM, nilông, ôtô |
| Token BPE trung bình / âm tiết (mọi âm tiết có chữ hoặc số) | 4,01 | 3,89 |
| … âm tiết có ký tự ngoài ASCII / âm tiết chỉ ASCII | 4,35 / 1,62 | 4,28 / 1,69 |
| Tỉ lệ âm tiết có ký tự ngoài ASCII | 87,5 % | 85,1 % |
| Phân bố 1 / 2 / 3 / 4+ token | 5,5 / 13,1 / 22,6 / 58,8 % | 5,7 / 14,5 / 23,3 / 56,5 % |

Cách đọc:

- **Rủi ro ngược của FSM chặt là khoảng 1 % âm tiết.** Đây là cỡ phần văn bản
  sẽ bị phá nếu FSM không có lối thoát tên riêng. Con số cùng bậc với ngưỡng
  trần 2 điểm % gợi ý trong `lo_trinh.md`, nên bước 4 **bắt buộc** phải có lối
  thoát. Nhiều trường hợp nằm ngoài từ điển là viết tắt (UBND, TP, VN) hoặc
  chính tả cũ (ôtô, nilông), không phải tên riêng.
- `allow_domain_tokens` gần như không ảnh hưởng trên bộ dữ liệu báo chí này
  (chênh ≤ 0,15 điểm %).
- Phân mảnh theo tần suất trên văn bản thật **nặng hơn** mức theo từ điển: hơn
  một nửa số âm tiết bị cắt thành 4 token trở lên. Đây là số của riêng Florence-2.
  Phần so chéo tokenizer (B5) còn chờ file tokenizer của Qwen2-VL, mBART/XLM-R
  và PhoBERT (xem D5).

### D5. Việc còn lại

1. Bước 3b: chạy `classify_errors_oracle.py --pred_jsonl <λ=0 trên val> --split_jsonl data/splits/val.jsonl`
   để lấy Δ_oracle chính thức, rồi áp quy tắc rẽ nhánh đã chốt ở bước 2b.
2. So chéo tokenizer (B5): cần commit thêm `tokenizer.json` của các tokenizer
   đối chứng vào `data/tokenizers/<tên>/`, rồi mở rộng
   `analysis/verify_bpe_fragmentation.py` cho mức tần suất.
3. Kiểm tra độ chính xác của mảnh byte so với ASCII trong lượt λ=0 đầu tiên,
   theo ngưỡng báo động đã chốt ở C3.2: val có U+FFFD > 1 % số dòng, hoặc mảnh
   byte kém ASCII > 15 điểm %.
