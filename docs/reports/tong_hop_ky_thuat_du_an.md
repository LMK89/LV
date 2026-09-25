# Tổng hợp kỹ thuật dự án LV (nguồn tham chiếu duy nhất)

Đề tài: **Ràng buộc cấu trúc âm tiết tiếng Việt cho OCR bằng VLM dùng byte-level
BPE: phân tích giới hạn và giải pháp**. Repo: `LMK89/LV`.
Trạng thái tính đến 25/09/2026: đã xong Bước 1a, 1b, 2 và 2b. Bước 3 đã có code
và dry-run, **chưa có lượt train nào trên GPU**.

Tài liệu này tự đủ: đọc riêng nó là nắm được dự án. Mỗi mục ghi rõ đâu là
**đã kiểm chứng**, đâu là **giả thuyết**, đâu là **dự kiến**. Khi tài liệu này
mâu thuẫn với code hoặc với `docs/lo_trinh.md`, thì `lo_trinh.md` (quy tắc đã
chốt) và code là đúng. Khi đó cần sửa lại tài liệu này.

Vai trò trong dự án:
- Claude Code: kiến trúc và thiết kế học thuật, lưu ở `docs/designs/`.
- Antigravity: viết code, test và merge.
- Máy local không có GPU. Train chạy trên GPU riêng.

---

## 0. Bản đồ repo

| Đường dẫn | Vai trò |
|---|---|
| `src/vocr/models/loader.py` | Điểm nạp model duy nhất. Dùng Florence-2 bản native của transformers (`florence-community/Florence-2-base`) và gắn DoRA qua PEFT. Tự từ chối merge nếu adapter chứa `lm_head` mà trọng số đang dùng chung |
| `src/vocr/nesy/syllables.py` | Từ điển 7.184 âm tiết, mở rộng thêm kiểu đặt dấu cũ/mới. Chứa `is_valid_syllable`, `could_start_valid_syllable`, `context_forgiven_rows` |
| `src/vocr/nesy/loss.py` | `compute_eligible_token_ids`, `compute_invalid_token_ids`, `build_invalid_mask`, `NeSyTrainer` (có log `eval_ce`) |
| `src/vocr/nesy/constraints.py` | `VietnamesePhonologyLogitsProcessor` (chế độ `token` và `automaton`, ràng buộc cấp chuỗi). **Chưa phải** FSM cấp byte của Bước 4 |
| `src/vocr/eval/metrics.py` | CER/WER theo NFC. Số chính là CER **toàn corpus** (micro) |
| `src/vocr/eval/oracle.py` | Căn chỉnh âm tiết, phân loại lỗi, CER Oracle, bootstrap theo tài liệu |
| `src/vocr/eval/significance.py` | Bootstrap ghép cặp theo cụm (`--cluster_by document\|article\|line`) |
| `analysis/classify_errors_oracle.py` | Chạy phân loại lỗi và CER Oracle từ dòng lệnh (`--dry-run`, `--pred_jsonl`, `--split_jsonl`) |
| `scripts/train.py`, `scripts/evaluate.py`, `scripts/sweep.py` | Train một lượt, đánh giá một cấu hình, điều phối 15 lượt |
| `configs/dora.yaml`, `train.yaml`, `sweep_step3.yaml`, `train_dryrun.yaml`, `augment.yaml` | Cấu hình |
| `data/splits/{train,val,test}.jsonl`, `split_report.md` | Dữ liệu đã chia. Mỗi bản ghi có `image, prefix="<OCR>", suffix, document, article` |
| `data/tokenizers/` | Tokenizer Florence-2 offline (`tokenizer.json`, `vocab.json`, `merges.txt`, …) |
| `docs/lo_trinh.md` | Lộ trình và **quy tắc rẽ nhánh đã chốt** |
| `docs/debug_fffd.md`, `docs/designs/step2_error_analysis.md`, `docs/designs/step3_lambda_sweep.md` | Chẩn đoán Bước 1b, thiết kế Bước 2 và Bước 3 |

Chạy test: `PYTHONPATH=src python -m pytest -q` (hoặc `python scripts/run_tests.py`).
Test mask cần `torch` và `transformers`.

---

## 1. Tổng quan đề tài và kiến trúc nền

### 1.1. Bài toán

OCR dòng văn bản tiếng Việt bằng VLM. Đầu vào là ảnh một dòng và prompt `<OCR>`,
đầu ra là chuỗi văn bản.

Câu hỏi nghiên cứu: tokenizer byte-level BPE (thiết kế cho tiếng Anh) cắt vụn âm
tiết tiếng Việt thì gây lỗi gì, và ràng buộc cấu trúc âm tiết (lúc train hoặc lúc
giải mã) sửa được tới đâu.

### 1.2. Mô hình

- **Florence-2-base** (khoảng 0,23 tỉ tham số): encoder thị giác DaViT, rồi bộ
  chiếu `image_projection`, rồi mô hình ngôn ngữ **encoder–decoder kiểu BART**
  (6+6 lớp, d_model=768). Đặc trưng ảnh cùng prompt đi vào encoder, decoder sinh
  văn bản.
- Dùng bản **native** của transformers (`Florence2ForConditionalGeneration`, cần
  transformers ≥ 4.56), không dùng `trust_remote_code`.
- Tinh chỉnh bằng **DoRA** (`configs/dora.yaml`): r=24, α=48, dropout 0,05,
  rsLoRA, khởi tạo gaussian, `task_type: CAUSAL_LM` (giữ như cũ để không đổi ngữ
  nghĩa). Các module được gắn adapter:
  - text tower: `q_proj`, `k_proj`, `v_proj`, `out_proj`, `fc1`, `fc2`
  - vision tower: `qkv`, `proj`
  - bộ chiếu: `image_projection`
  - **không gắn `lm_head`**.

### 1.3. Tokenizer và hiện tượng phân mảnh (đã đo)

- Byte-level BPE (`BartTokenizerFast`), `len(tokenizer)` = **51.290**. Bảng
  `lm_head` có 51.328 hàng: 38 hàng dư không ứng với token nào và phải bị chặn khi
  giải mã (`constraints.py` đã làm).
- Một ký tự có dấu dài 2–3 byte UTF-8 và thường bị cắt thành từng byte. Ví dụ
  `"Việt Nam"` → `<s> Vi á » ĩ t ĠNam </s>`. Có 352 token là **mảnh byte**, tức
  decode riêng ra chuỗi chứa U+FFFD.
- Trên nhãn val/test, trung bình **khoảng 4 token cho một âm tiết**:
  - âm tiết có ký tự ngoài ASCII: khoảng 4,3 token
  - âm tiết chỉ ASCII: khoảng 1,6 token
  - âm tiết bị cắt thành ≥ 4 token: 57–59 %
  - âm tiết có ký tự ngoài ASCII chiếm 85–88 % tổng số âm tiết
- Tokenizer tự nó không lỗi: `decode(encode(x)) == x` đúng trên 936/936 dòng train.
- Đã kiểm trên nhãn val/test: khoảng **1 % âm tiết nằm ngoài từ điển**, chủ yếu là
  viết tắt (UBND, TP, VN) và chính tả cũ (ôtô, nilông). Đây là phần FSM chặt sẽ phá
  nếu không có lối thoát.

### 1.4. Lỗi U+FFFD ở v1: nguyên nhân và cách xử lý

- **Triệu chứng (v1, đã đo):** Stage 1 cho 199/200 dòng có U+FFFD, và 26,7 % ký tự
  đầu ra là U+FFFD. Model gốc zero-shot thì 0/200.
- **Nguyên nhân (giả thuyết mạnh, CHƯA xác nhận trên GPU):**
  - DoRA gắn vào `lm_head`.
  - `lm_head` dùng chung trọng số (`tie_word_embeddings=True`) với embedding
    `shared`, tức embedding của **cả encoder lẫn decoder**.
  - Khi đánh giá, `merge_and_unload()` ghi trọng số đã gộp vào `lm_head`, nên
    embedding đầu vào cũng đổi theo.
  - Lúc train (chưa gộp) embedding vẫn là bản gốc, nên suy luận lệch khỏi train.
  - Bằng chứng gián tiếp: PEFT đã cảnh báo về `ensure_weight_tying` ngay trong log
    train của v1.
  - Việc xác nhận là cổng **G0** (`scripts/debug_fffd.py model`, khoảng 5 phút GPU).
- **Cách xử lý (đã áp dụng):**
  - Bỏ `lm_head` khỏi `target_modules`, tiết kiệm khoảng 1,30 triệu tham số
    (= 24·(768+51.290) + 51.290).
  - `loader.load_lora_model` từ chối merge nếu adapter vẫn có `lm_head`.
  - Kiểm tra dry-run F4.5 dự kiến so logits merged với unmerged.
- **Rủi ro còn lại:** khi không có `lm_head` trong adapter, xác suất ưu tiên của
  các mảnh byte hiếm phải học qua hidden state. Rủi ro này được theo dõi bằng
  ngưỡng báo động ở cổng G2 (mục 6.4).
- **Hệ quả cho luận văn:** mọi số liệu của v1 (CER 59,6 %…) đều **không dùng
  được**, vì hai lý do: lỗi merge, và tập chia của v1 bị rò rỉ.

---

## 2. Dữ liệu và cách chia độc lập tài liệu

- 1.350 ảnh dòng văn bản (`data/raw/labels.jsonl`). Mỗi dòng gắn `document` (mã
  tài liệu) và `article` (mã bài).
- Cùng một bài có thể được nhiều tài liệu chép lại (48 câu trùng nguyên văn giữa các
  tài liệu, luôn cùng mã bài). Vì vậy **chia theo tài liệu vẫn rò rỉ**, và phải chia
  theo **mã bài**.
- Tập chia cố định, seed 165, tỉ lệ mục tiêu 70/15/15:

| Tập | Số dòng | Tỉ lệ | Số bài | Số tài liệu (cụm bootstrap) |
|---|---|---|---|---|
| train | 936 | 69,3 % | 31 | 182 |
| val | 202 | 15,0 % | 6 | **31** |
| test | 212 | 15,7 % | 7 | **33** |

- Rò rỉ giữa mọi cặp tập đều bằng 0 (bài, tài liệu, câu trùng). Tên dùng trong luận
  văn: **"Document-independent evaluation"**. Đã bác bỏ phương án K-fold.
- **Hạn chế:** tập test nhỏ (7 bài), nên hiệu ứng khoảng 1–2 điểm CER khó đạt ý
  nghĩa thống kê. Cần báo cáo hiệu ứng nhỏ nhất phát hiện được (MDE), hoặc mở rộng
  dữ liệu.
- **Augment:** `configs/augment.yaml` sinh `train_augmented.jsonl` gồm bản gốc
  cộng 3 bản biến đổi mỗi ảnh, seed 42, **chỉ từ train**. File này chưa tồn tại và
  sẽ được sinh trên máy GPU (cổng G1).

---

## 3. NeSy Loss

### 3.1. Công thức (bản dùng cho Bước 3)

$$\mathcal{L} = \mathcal{L}_{CE} + \lambda\,\frac{\sum_t m_t\, s_t}{\sum_t m_t + \epsilon},\qquad
s_t = \sum_{v \in \mathcal{M} \setminus \{y_t\}} p_t(v)$$

- $\mathcal{M}$ là tập token bị phạt (mask rule hoặc random).
- $y_t$ là token đúng của nhãn, **không bao giờ bị phạt**, để NeSy không chống lại
  CE khi nhãn là tên riêng hoặc viết tắt.
- $m_t = 1$ khi và chỉ khi vị trí $t$ có nhãn (khác -100) **và** không được tha thứ.
- Gradient đi qua softmax, rồi logits, rồi adapter.

### 3.2. Các tập token (đã kiểm bằng test)

| Tập | Định nghĩa | Kích thước |
|---|---|---|
| vocab | `len(tokenizer)` | 51.290 |
| eligible | `decode([t])` không rỗng, không toàn khoảng trắng, không chứa U+FFFD. Vì vậy đã loại token đặc biệt và 352 mảnh byte | **49.885** |
| rule | các token trong eligible có `not is_valid_syllable(decode([t]))` | **45.775** (khoảng 89 % vocab, phần lớn là subword tiếng Anh) |
| random (đối chứng) | 45.775 token lấy mẫu **trong eligible**, `mask_seed = seed` của lượt chạy | 45.775 |

### 3.3. Cửa sổ tha thứ subword (W = 5)

- Một vị trí nhãn được tha nếu nó nằm trong một dải 2–5 token liên tiếp thỏa hai
  điều kiện: dải có chứa token thuộc `rule`, và decode cả dải ra chỉ gồm âm tiết
  hợp lệ.
- **Vị trí tha thứ luôn tính bằng tập `rule`**, cho cả hai điều kiện. Đây là tính
  chất của nhãn, không phải của mask.

### 3.4. Vì sao phải sửa đối chứng random (đã đo trên 936 dòng train)

| Mask | Jaccard với rule | Phạt EOS | Vị trí được tha | Token đúng bị phạt (không được tha) |
|---|---|---|---|---|
| rule | 1,00 | không | 77,4 % | 2,2 % |
| random **cũ** (lấy mẫu trên toàn vocab, vị trí tha tính theo chính nó) | 0,81 | **có** | **96,1 %** | 3,8 % |
| random **mới** (lấy mẫu trong eligible, dùng vị trí tha của rule, bỏ token đúng) | khoảng 0,85 | không | = rule | 0 % (theo định nghĩa) |

Hai mask vẫn trùng khoảng 85 %. Vì vậy **diễn giải được đăng ký trước**: nếu rule
thắng λ=0 nhưng không thắng random, thì NeSy chỉ hoạt động như một bộ điều chuẩn
chung, không phải nhờ tri thức âm tiết.

### 3.5. Chọn checkpoint

- `NeSyTrainer.evaluate` log `eval_ce`, là CE thuần, không có penalty.
- `metric_for_best_model: eval_ce` áp dụng cho **mọi** λ. Trước đây `eval_loss`
  gồm cả penalty, nên các λ bị chọn checkpoint theo thước đo khác nhau.

### 3.6. Tương tác với bộ kiểm tra âm tiết

- `is_valid_syllable` từ chối mọi chuỗi có U+FFFD.
- `could_start_valid_syllable` cho phép một dãy U+FFFD **ở cuối** chuỗi (ký tự
  nhiều byte đang viết dở), nhưng từ chối nếu U+FFFD nằm giữa chuỗi.
- Nếu đảo ngược một trong hai quy tắc này:
  - chế độ `token` sẽ không sinh được các từ bắt đầu bằng ở, ấy, ạ, ứng…;
  - mask rule sẽ phạt cả 352 mảnh byte.

---

## 4. CER Oracle và phân loại lỗi (`src/vocr/eval/oracle.py`)

### 4.1. Căn chỉnh

- Đơn vị âm tiết là một khối `\S+`, giữ vị trí ký tự trong chuỗi gốc. Mọi chuỗi
  được chuẩn hóa NFC.
- Quy hoạch động Levenshtein với chi phí nguyên:
  - chèn/xóa: 1000;
  - thay thế: 1000 × khoảng cách ký tự chuẩn hóa;
  - tách/gộp k:1 hoặc 1:k (k ≤ 3): `1 + 1000·d`, chỉ cho phép khi d ≤ 1/3, với d
    là khoảng cách giữa chuỗi nối lại và phía bên kia.
- Khi hòa điểm, ưu tiên thay thế, rồi tách/gộp, rồi xóa, rồi chèn.

### 4.2. Phân loại

`inv(x)` nghĩa là x chứa U+FFFD hoặc `not is_valid_syllable(x)`.

| Nhóm | Điều kiện | Ý nghĩa với FSM |
|---|---|---|
| **a1** | thay thế, `inv(dự đoán)`, nhãn hợp lệ | FSM chặt sửa được về nguyên tắc |
| **a2** | thay thế, `inv(dự đoán)` **và `inv(nhãn)`** (tên riêng, viết tắt, từ ngoại) | Chỉ sửa được nếu FSM có lối thoát |
| **b1** | thay thế, dự đoán hợp lệ, chỉ khác dấu (thanh, mũ/móc/trăng, đ↔d). Nhãn phụ `detail=tone` khi chỉ khác 5 dấu thanh | Ngoài tầm FSM |
| **b2** | thay thế, dự đoán hợp lệ, khác chữ cái | Ngoài tầm FSM |
| **c** | `c_del`, `c_ins_valid`, `c_ins_invalid` | Ngoài tầm (trừ việc xóa rác, chỉ tính ở cận trên) |
| seg | tách/gộp: `seg_a1` / `seg_a2` khi có ít nhất một âm tiết dự đoán không hợp lệ, `seg_valid` khi tất cả hợp lệ | `seg_a1` được **tính vào trần chính** |

Nhãn phụ chỉ để thống kê: `b_case` (chỉ khác hoa/thường), `b_punct` (chỉ khác dấu
câu dính kèm).

Lưu ý: nhóm **a2 không có nghĩa** là "ngoài từ điển nhưng cấu trúc hợp lệ". Nó
nghĩa là *nhãn* nằm ngoài từ điển, còn dự đoán không hợp lệ.

### 4.3. Trần lý thuyết

$$\Delta_{\text{oracle}} = \text{CER}_{\text{raw}} - \text{CER}_{\text{oracle}}$$

`CER_oracle` là CER sau khi thay mọi phần dự đoán thuộc **a1 + seg_a1** bằng phần
nhãn tương ứng.

- Việc thay được làm **tại đúng vị trí ký tự**, phần còn lại của chuỗi giữ nguyên.
- **Chốt chặn:** với mỗi dòng, edit của oracle = min(edit trước, edit sau khi sửa).
  Oracle có quyền không can thiệp. Ví dụ `chúng ta ăn` → `chúngta ăn`: sửa ngây thơ
  cho 3 edit, trong khi gốc chỉ có 1.
- Hai cận trên:
  - `oracle_a`: sửa thêm a2 và seg_a2.
  - `oracle_a_ins`: xóa thêm các `c_ins_invalid`.
- Báo cáo kèm theo:
  - số dòng mà chốt chặn phải can thiệp (`lines_worse_*`);
  - tỉ lệ âm tiết nhãn ngoài từ điển;
  - số âm tiết "đã đúng nhưng ngoài từ điển" (FSM chặt sẽ làm hỏng);
  - tỉ lệ lỗi theo số token BPE của âm tiết nhãn.
- Khoảng tin cậy 95 %: bootstrap chọn lại tài liệu, tính lại CER corpus.
- Kiểm chứng trên nhiễu tổng hợp (1.010 dòng): chốt chặn chỉ phải can thiệp ở
  2 dòng, thời gian khoảng 13 giây trên CPU.

---

## 5. Giao thức thống kê và quy tắc rẽ nhánh (đăng ký trước)

Nguồn chính thức là `docs/lo_trinh.md`. Ngưỡng chốt ngày 25/09/2026, commit
`18488b4`. Phần H4 được bổ sung **trước khi có bất kỳ số liệu Bước 3 nào**.

### 5.1. Trên val (quyết định)

- **Δ_oracle** tính cho a1 + seg_a1 trên val, lấy trung bình 3 seed của λ=0.
- Khoảng tin cậy 95 %: bootstrap theo tài liệu (31 cụm). Mỗi lần bootstrap tính Δ
  cho từng seed rồi lấy trung bình 3 seed.
- **λ\*** ∈ {0,1; 0,5} là giá trị có CER corpus trung bình 3 seed thấp hơn của
  mask rule.

### 5.2. Trên test (chạy một lần)

- Hai so sánh, hiệu chỉnh Holm:
  1. `rule(λ*)` với `lam0`
  2. `rule(λ*)` với `random(λ*)`
- Bootstrap ghép cặp theo tài liệu (33 cụm) trên **CER corpus**, một phía,
  10.000 lần. Dự đoán của 3 seed được gộp ở mức số edit và số ký tự của từng dòng.

### 5.3. Quy tắc rẽ nhánh

| Điều kiện | Nhánh |
|---|---|
| NeSy (rule) tốt hơn λ=0 **và** tốt hơn random, p < 0,05 (Holm) | Giữ NeSy trong phần giải pháp |
| Cận trên khoảng tin cậy 95 % của Δ_oracle < **2,0** điểm % CER | Trần thấp: dồn vào phân tích, so chéo tokenizer, mở rộng vocab |
| Δ_oracle ≥ 2,0 và FSM lấy được ≥ **70 %** của Δ_oracle | Đóng góp chính là FSM + phân tích, bỏ DPO |
| Δ_oracle ≥ 2,0 và FSM lấy được < 70 % | Làm DPO trên chuỗi tự sinh (tiêu chí chính là CER), bắt buộc có đối chứng DPO chỉ tối ưu CER |
| **H4:** Δ_oracle < 2,0 nhưng cận trên ≥ 2,0 | Chưa đủ bằng chứng rằng trần thấp, nên đi theo nhánh "≥ 2,0" (đo FSM ở Bước 4) và báo cáo rõ khoảng tin cậy |

Diễn giải đăng ký trước cho NeSy: nếu rule thắng λ=0 nhưng không thắng random, kết
luận là NeSy chỉ đóng vai điều chuẩn chung. Kết luận này vẫn được công bố.

---

## 6. Thiết kế thí nghiệm Bước 3 (`exp/04-quet-lambda`)

Thiết kế đầy đủ nằm ở `docs/designs/step3_lambda_sweep.md`.

### 6.1. Ma trận thí nghiệm

5 điều kiện × seed {1, 2, 3} = **15 lượt**, chạy theo thứ tự **seed-major** (xong
5 điều kiện của seed 1 rồi mới sang seed 2):

| Điều kiện | λ | mask |
|---|---|---|
| `lam0` | 0 | none |
| `lam0.1_rule` / `lam0.1_rand` | 0,1 | rule / random |
| `lam0.5_rule` / `lam0.5_rand` | 0,5 | rule / random |

Run ID có dạng `{condition}_s{seed}`, ví dụ `lam0.1_rule_s2`.

### 6.2. Cấu hình

- `train.yaml`:
  - dữ liệu `train_augmented.jsonl`
  - batch 1 × gradient accumulation 8, lr 1e-4, cosine, warmup 5 %, 5 epoch
  - bf16 (train.py dừng nếu GPU không hỗ trợ)
  - eval mỗi 100 step, lưu mỗi 200 step, `save_total_limit: 2`
  - `load_best_model_at_end`, chọn theo `eval_ce`, early stopping patience 15
  - `max_label_length` 768
- `train_dryrun.yaml`: 4 mẫu train, 4 mẫu eval, `max_steps: 1`, fp32,
  `max_label_length` 64, dùng `train.jsonl`.
- `sweep_step3.yaml`:
  - khai báo ma trận ở 6.1, `output_root: outputs/step3`
  - eval trên val, `num_beams: 3`, `max_new_tokens: 256`
  - `gates`: `fffd_line_rate_max: 0.01`, `byte_vs_ascii_acc_gap_max: 0.15`

### 6.3. Điều phối (`scripts/sweep.py`)

- Các chế độ: `--plan`, `--status`, `--run [--only <run_id>]`, `--dry-run`.
- Bỏ qua lượt đã có file `DONE`.
- Mỗi lượt ghi vào `outputs/step3/<run_id>/`: `run_info.json` (args, commit, cấu
  hình, số tham số train), `train.log`, `checkpoint-*`, `final_adapter/`, `DONE`.
- Không commit checkpoint và adapter. Chỉ commit manifest, `run_info`, metrics và
  kết quả eval/oracle.

**Hiện thực so với thiết kế (tính đến commit `859ba54`).** Những điểm dưới đây
thiết kế đã nêu nhưng code chưa có. Cần làm xong trước cổng G1:

| Thiết kế | Hiện trạng |
|---|---|
| Mỗi lượt có 3 pha: train, `evaluate.py` trên val, oracle | Mới có pha train |
| `manifest.csv` (commit, sha256 cấu hình và dữ liệu, version, GPU, CER val…) | Chưa ghi |
| Lượt lỗi thì ghi `failed` rồi chạy tiếp; lượt dở thì chạy tiếp bằng `--resume` | Lỗi thì dừng cả sweep, chưa tự resume |
| Commit bẩn thì từ chối chạy (trừ khi có `--allow-dirty`) | Chỉ cảnh báo |
| Callback đo `byte_acc` / `ascii_acc` lúc eval (để áp cổng G2) | Chưa có |
| Dry-run F-tiny (model thu nhỏ, trọng số ngẫu nhiên, không tải weight) | `train.py` vẫn nạp Florence-2-base thật qua `load_model_and_processor` |
| Kiểm tra dry-run F4.5: logits merged với unmerged khớp nhau | Chưa có |

### 6.4. Các cổng kiểm soát trên GPU

| Cổng | Việc | Điều kiện qua |
|---|---|---|
| G0 | `scripts/debug_fffd.py model` trên checkpoint v1 để xác nhận giả thuyết lỗi merge | Ghi kết quả vào `docs/debug_fffd.md` |
| G1 | Sinh `train_augmented.jsonl` và ghi sha256. Chạy dry-run với model thật, `max_steps=5` | Mọi assert dry-run đều đạt |
| G2 | Chạy trọn `lam0_s1` và đo thời gian | Val có U+FFFD ở ≤ 1 % số dòng **và** `ascii_acc − byte_acc` ≤ 15 điểm %. Không qua thì **dừng lại**, họp chọn phương án dự phòng (`trainable_token_indices` cho các hàng embedding của mảnh byte) |
| G3 | Chạy 14 lượt còn lại theo thứ tự seed-major | — |
| G4 | Bước 3b: Δ_oracle chính thức trên val, áp quy tắc rẽ nhánh | — |
| G5 | Chạy test một lần cho hai so sánh ở 5.2 | — |

### 6.5. Sau Bước 3 (dự kiến)

- **Bước 4:** FSM chính xác cấp byte, không cho qua U+FFFD vô điều kiện, có lối
  thoát cho tên riêng. Đo phần FSM lấy được so với Δ_oracle.
- **Bước 5:** gặp GVHD với số liệu.
- **Bước 6** (DPO): chỉ làm nếu quy tắc rẽ nhánh cho phép.
- Việc còn treo:
  - so chéo tokenizer (Qwen2-VL, mBART/XLM-R, PhoBERT, bảng ký tự của VietOCR),
    cần file tokenizer tương ứng;
  - khóa phiên bản `peft`;
  - bootstrap trên CER corpus trong `significance.py`.
