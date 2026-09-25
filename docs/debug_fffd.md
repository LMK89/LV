# Bước 1b — Chẩn đoán U+FFFD trong output Stage 1 (v1)

Script: `scripts/debug_fffd.py` (ba lệnh con `tokenizer`, `model`, `preds`).
Dữ liệu v1 đối chiếu: `LMK89/CS2225.K2025.Florence2-Vietnamese@d977fd4`
(`outputs/results/eval_2. Stage1_NeSyLoss_20260831_035128.jsonl`, `explog/train_v2.log`,
`outputs/checkpoints/stage1/final_adapter/adapter_config.json`).

## Kết luận

**Nguyên nhân gần như chắc chắn là lỗi pipeline lúc suy luận, không phải do model
học kém hay do NeSy:** DoRA được gắn vào `lm_head`, mà `lm_head` của Florence-2
**dùng chung trọng số** (tied) với embedding token của decoder. Lúc đánh giá,
`load_lora_model(..., merge=True)` gọi `merge_and_unload()`, ghi trọng số đã gộp
vào `lm_head`, và vì dùng chung nên **embedding đầu vào của decoder cũng bị đổi
theo**. Khi train (adapter chưa gộp) thì embedding đầu vào vẫn là bản gốc. Hai
lúc lệch nhau, nên `eval_loss` (đo khi train) đẹp mà output (đo sau khi gộp)
hỏng.

Trạng thái: **giả thuyết mạnh, chưa xác nhận trên model**. Máy chạy phân tích này
không tải được weight Florence-2 (huggingface.co bị chặn), còn adapter v1 trong
git chỉ là con trỏ LFS. Lệnh xác nhận nằm ở mục "Cách xác nhận" bên dưới, chạy
khoảng 5 phút trên GPU.

## 1. Tokenizer: không phải nguyên nhân (đã chạy)

`python scripts/debug_fffd.py tokenizer --model <final_adapter của v1>`. Đây là
đúng tokenizer Florence-2 (`BartTokenizerFast`, 51.290 token) lưu kèm checkpoint.
Chạy trên `data/splits/train.jsonl`:

| Kiểm tra | Kết quả |
|---|---|
| Nhãn không ở dạng NFC | 0/936 |
| Round-trip `decode(encode(x)) == x` hỏng | **0/936** |
| Decode **cả chuỗi** có U+FFFD | **0/936** |
| Decode **từng token** rồi nối có U+FFFD | **932/936 (99,6 %)** |
| Token là mảnh byte (tự nó không phải UTF-8 hợp lệ) | 26.675/56.759 (47,0 %) |

Ví dụ: `"Việt Nam"` được cắt thành `<s> Vi á » ĩ t ĠNam </s>`. Chữ `ệ` (3 byte
UTF-8) bị tách thành 3 token byte.

- Tokenizer mã hóa và giải mã đúng; chuỗi nhãn không có gì bất thường.
- Tỉ lệ 99,6 % của cách "decode từng token" trùng gần như tuyệt đối với 199/200
  của v1, nên đây từng là nghi phạm số 1. **Nghi phạm này đã bị loại bằng code:**
  `evaluate.py` của cả v1 (dòng 89) lẫn LV đều dùng
  `processor.batch_decode(generated_ids)` trên cả chuỗi. Vậy U+FFFD trong v1 là
  do **chính dãy byte model sinh ra** không phải UTF-8 hợp lệ.
- Gần một nửa số bước sinh là mảnh byte. Chỉ cần sai **cấu trúc** ở một bước
  (byte đầu không có byte nối, hoặc byte nối không có byte đầu) là ra U+FFFD.

## 2. Output v1 có dấu hiệu lệch train/suy luận (đã chạy)

`python scripts/debug_fffd.py preds --pred_jsonl "eval_2. Stage1_NeSyLoss_…jsonl"`:

| | Nhãn | Dự đoán Stage 1 |
|---|---|---|
| Dòng có U+FFFD | — | 199/200 |
| Ký tự ASCII | 78,0 % | 67,8 % |
| Ký tự ngoài ASCII **hợp lệ** (chữ có dấu) | 22,0 % | **5,4 %** |
| U+FFFD | 0 % | **26,7 %** |

Florence-2 gốc (zero-shot, cùng code, không có adapter): 0/200 dòng có U+FFFD.

Ví dụ: `Có khi chúng lừa lúc chủ gà…` → `óhiú chng� l�aú ch�ng�a�…`

Phần ASCII vẫn đọc được gần đúng, nhưng gần như **mọi chữ có dấu đều vỡ cấu trúc
UTF-8**. Với `eval_loss = 0,60` (teacher forcing), model đã phải học được quy luật
rất đơn giản "sau byte đầu là byte nối". Model mà quên quy luật này ở 1/4 số ký tự
khi sinh tự do thì khó giải thích bằng "học chưa đủ". Nó khớp với giả thuyết lệch
embedding: DoRA scale lại từng hàng của `lm_head`, và các hàng bị đổi nhiều nhất là
những token hay được train làm output, tức chính các mảnh byte tiếng Việt. Sau khi
merge, *embedding đầu vào* của các token này bị đổi, nên decoder "không nhận ra"
mình vừa sinh byte đầu của một ký tự nhiều byte. Phần "hàng bị đổi nhiều nhất" là
suy luận, mục 2 của lệnh `model` sẽ đo trực tiếp.

## 3. Bằng chứng cấu hình (đã đọc)

- `adapter_config.json` của v1: `use_dora: true`, `target_modules` chứa
  **`lm_head`**, `ensure_weight_tying: false`.
- `explog/train_v2.log`, dòng 2, là cảnh báo của PEFT:
  > Model has `tie_word_embeddings=True` and a tied layer is part of the adapter,
  > but `ensure_weight_tying` is not set to True. This can lead to complications,
  > for example when merging the adapter …
- `load_lora_model(checkpoint, device, merge=True)` là mặc định, và
  `evaluate.py` của v1 gọi đúng hàm này nên **luôn merge**.
- **LV vẫn giữ nguyên cả ba điều trên:** `configs/dora.yaml` có `lm_head`,
  `loader.py` mặc định `merge=True`. Nếu không sửa, bước 3 sẽ lặp lại lỗi này
  với mọi λ.

## 4. BOS và decoder start (một phần)

- Nhãn (`processor.tokenizer(suffix)`) có dạng `[0=<s>, …, 2=</s>]`. Tokenizer
  có bos/eos/pad = 0/2/1, đúng quy ước BART (decoder bắt đầu bằng `</s>`=2 và
  sinh `<s>` trước).
- Giá trị `decoder_start_token_id`, `forced_bos_token_id` trong `config` và
  `generation_config` của bản native **chưa đọc được ở đây** (cần tải model).
  Mục 1 của lệnh `model` in ra các giá trị này.
- Kể cả khi hai giá trị này lệch nhau, hậu quả thường là thừa hoặc thiếu token
  đầu câu, không làm vỡ cấu trúc byte ở giữa dòng. Vì vậy đây khó là nguyên nhân
  chính.

## Cách xác nhận (cần GPU, khoảng 5 phút)

```bash
python scripts/debug_fffd.py model \
    --checkpoint <v1>/outputs/checkpoints/stage1/final_adapter \
    --jsonl data/splits/train.jsonl --only_v1_train --n 20 --out outputs/debug_fffd_model.md
```

Lệnh này in 4 bảng: (1) token đặc biệt và cấu hình sinh; (2) embedding trước và
sau `merge_and_unload()`; (3) loss và độ chính xác teacher forcing, merged so với
unmerged; (4) sinh greedy trên ảnh train, merged so với unmerged, decode cả chuỗi
so với từng token.

Cách đọc kết quả:

| Quan sát | Kết luận |
|---|---|
| Bảng 2: embedding decoder đổi (max \|Δ\| > 0), **và** bảng 4: unmerged sạch còn merged có U+FFFD | **Xác nhận lỗi merge + tied `lm_head`** |
| Cả unmerged lẫn merged đều có U+FFFD trên ảnh **train** | Không phải merge. Xem lại bảng 1 (BOS/decoder start) và bảng 3 |
| Unmerged sạch trên ảnh train nhưng test vẫn hỏng | Model học được nhưng tổng quát kém (dữ liệu ít), không phải lỗi pipeline |

`--only_v1_train` chỉ lấy những ảnh checkpoint v1 thật sự đã thấy lúc train
(702/936 dòng của `train.jsonl` mới thuộc train v1), nên đây đúng là phép thử
"decode trên ảnh model đã thấy".

## Đề xuất sửa (chưa áp dụng, chờ xác nhận)

1. **`load_lora_model`: không merge khi adapter có module dùng chung trọng số**
   (hoặc đổi mặc định thành `merge=False`). Suy luận chậm hơn một chút nhưng
   giống hệt lúc train. Nên thêm kiểm tra tự động: nếu `lm_head` nằm trong
   `target_modules` và `tie_word_embeddings=True` thì từ chối merge.
2. **Cho bước 3, chọn một trong hai phương án, rồi giữ cố định cho mọi cấu hình:**
   - Giữ `lm_head` trong DoRA và không bao giờ merge (giữ nguyên thiết lập v1).
   - Bỏ `lm_head` khỏi `target_modules`. Sạch hơn, nhưng đổi dung lượng adapter,
     nên cần thống nhất với GVHD.
3. Đánh giá lại checkpoint v1 với `merge=False` để biết Stage 1 v1 thật sự đạt
   bao nhiêu. Chỉ để tham khảo, vì tập chia v1 bị rò rỉ.

## Các vấn đề phụ phát hiện thêm (không gây U+FFFD)

- `task_type: CAUSAL_LM` cho model encoder–decoder. Loại hợp lý là
  `SEQ_2_SEQ_LM`. Lúc train không thấy hỏng gì, nhưng nên sửa để tránh bọc sai
  lớp PEFT.
- `train_loss` trung bình 8,2 trong khi CE cuối khoảng 0,5 (`docs/lich_su_v1.md`).
  Vẫn cần kiểm tra `compute_loss` với gradient accumulation.

## Hệ quả cho luận văn

Toàn bộ số liệu Stage 1 của v1 (CER 59,6 %, 199/200 U+FFFD, "Stage 1 + ràng buộc
decode" 1091 %) đo **model sau khi bị merge hỏng**, không phản ánh DoRA hay NeSy.
Ngoài chuyện rò rỉ dữ liệu, đây là lý do thứ hai để không dùng các số đó.
