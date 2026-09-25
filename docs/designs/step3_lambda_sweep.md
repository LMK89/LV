# Thiết kế Bước 3: Quét λ × mask × seed (`exp/04-quet-lambda`)

Trạng thái: **BẢN THIẾT KẾ, chờ đồng thuận.** Vai trò: Claude thiết kế và review,
Antigravity viết code và chạy.
Phần cứng: máy local chỉ có CPU, chỉ dùng để dry-run. Train thật chạy trên GPU.

Tài liệu gồm: (A) các điểm cần chốt, (B) phát hiện mới (đã đo trên CPU),
(C) cấu hình, (D) điều phối 15 lượt chạy, (E) checkpoint và log, (F) dry-run
trên CPU, (G) quy trình trên GPU và cổng kiểm soát, (H) giao thức thống kê.

---

## A. Các điểm cần đồng thuận trước khi code

| # | Vấn đề | Đề xuất của Claude | Mức độ |
|---|---|---|---|
| A1 | Mask `random` hiện tại **không phải đối chứng công bằng** (số liệu ở B1) | Sửa 3 điểm: lấy mẫu trong tập token *đủ điều kiện*, dùng *chung vị trí tha thứ* với mask rule, và *không phạt token đúng của nhãn* | **Chặn**: nếu không sửa, phép so sánh rule với random vô nghĩa |
| A2 | `eval_loss` của các lượt λ>0 đang cộng thêm NeSy penalty | Chọn checkpoint tốt nhất bằng **CE thuần** (`eval_ce`) cho mọi λ | **Chặn**: các λ đang được chọn checkpoint theo thước đo khác nhau |
| A3 | Quy tắc rẽ nhánh có vùng chưa định nghĩa | Bổ sung (được phép vì chưa có số liệu Bước 3): Δ < 2 nhưng cận trên ≥ 2 thì **không kết luận được, coi như trần không thấp** | Cao |
| A4 | Δ_oracle lấy trên seed nào? | Trung bình trên 3 seed của λ=0. Khoảng tin cậy lấy từ bootstrap theo tài liệu, tính chung cho cả 3 seed | Cao |
| A5 | Tập dùng cho quyết định | **Val** để chọn λ và tính Δ_oracle. **Test chỉ chạy một lần** ở cuối, cho 2 so sánh đăng ký trước (H2) | Cao |
| A6 | `configs/train.yaml` trỏ tới `train_augmented.jsonl`, file này chưa tồn tại | Sinh file bằng `vocr.data.augment` (seed cố định) **trên máy GPU**, ghi lại hash của file | Trung bình |
| A7 | Ghi độ chính xác của mảnh byte để áp ngưỡng báo động C3.2 | Thêm một callback đo lúc eval (E3) | Trung bình |

---

## B. Phát hiện mới (đã đo trên CPU bằng tokenizer offline và `train.jsonl`)

### B1. Mask `random` hiện tại lệch so với mask rule ở 3 điểm

Cách đo: 936 dòng nhãn train, hàm `context_forgiven_rows` với `max_window=5`,
mask rule có 45.775 token. Các mask random được lấy mẫu giống cách
`build_invalid_mask` làm (cùng kích thước, chọn trên toàn vocab), thử 3 seed.

| Mask | Jaccard với rule | EOS (`</s>`=2) bị phạt? | Vị trí nhãn được tha | Vị trí có **token đúng của nhãn nằm trong mask** mà không được tha |
|---|---|---|---|---|
| rule | 1,000 | không | 77,4 % | 2,2 % |
| random trên toàn vocab (3 seed) | **0,805–0,806** | **có**, ở cả 3 seed (xác suất khoảng 0,89) | **96,1 %** | 3,7–3,8 % (có 104 vị trí EOS) |
| random chỉ trong tập đủ điều kiện | 0,847–0,848 | không | 89,5–90,2 % | 1,9–2,3 % |

Cách đọc:

1. **Vị trí bị phạt khác nhau tới 6 lần.** Hàm tha thứ nhận `self._invalid_token_ids`
   làm đầu vào. Với mask random, gần như cửa sổ nào cũng chứa một token bị
   "phạt", và cửa sổ đó lại decode ra âm tiết hợp lệ, nên **96 %** vị trí được
   tha. Kết quả là chỉ khoảng 3,9 % vị trí bị phạt, so với 22,6 % của mask rule.
   Như vậy "random" thực chất là một λ yếu hơn nhiều, chứ không phải "cùng mật
   độ nhưng khác tri thức". Nếu rule thắng random thì có thể chỉ vì rule điều
   chuẩn mạnh hơn.
2. **Mask random phạt cả EOS và token đặc biệt.** Nó phạt `</s>` ở mọi seed,
   nên kéo model về phía không kết thúc câu. Đây là một nguồn gây hại không liên
   quan tới tri thức âm tiết.
3. **Hai mask trùng nhau khoảng 81 %.** Dù sửa hết, hai mask chỉ khác nhau ở
   khoảng 15 % số token. Hiệu ứng rule so với random có thể nhỏ. Phải **đăng ký
   trước cách diễn giải** (H3).
4. **Ngay cả mask rule cũng phạt token đúng ở 2,2 % vị trí**: tên riêng, viết
   tắt, và các mảnh byte của chúng không được tha. Nghĩa là NeSy đang đẩy xác
   suất của **đáp án đúng** xuống, đi ngược với CE. Hiện tượng này khớp với
   khoảng 1 % âm tiết ngoài từ điển đo ở Bước 2.

### B2. Đề xuất sửa (A1). Ba thay đổi này cùng định nghĩa lại penalty

```
eligible  = {t : decode(t) không rỗng, không toàn khoảng trắng, không chứa U+FFFD}   # 49.885 token
rule      = {t ∈ eligible : not is_valid_syllable(decode(t))}                        # 45.775
random_s  = mẫu ngẫu nhiên |rule| phần tử trong eligible, seed s                     # đã loại token đặc biệt
forgiven  = context_forgiven_rows(labels, ..., rule_ids, ...)   # LUÔN tính bằng rule, cho cả hai mask
penalty_i = Σ_{t ∈ mask, t ≠ gold_i} p_i(t)                     # không bao giờ phạt token đúng
```

- *Vị trí tha thứ là tính chất của nhãn* (chỗ nào BPE cắt vụn một âm tiết hợp
  lệ), không phải tính chất của mask. Dùng chung vị trí tha thứ thì hai điều
  kiện chỉ còn khác nhau ở một điểm: **token nào bị đẩy xuống**.
- Bỏ token đúng ra khỏi penalty khiến NeSy chỉ lấy xác suất khỏi các *lựa chọn
  sai*, và không bao giờ chống lại CE. Việc này cũng giải quyết B1.4 cho tên
  riêng. Code: `penalty = (probs*mask).sum(-1) - probs.gather(-1, gold)*mask[gold]`.
  Vị trí có nhãn -100 thì bỏ qua như hiện tại.
- Test cần có, chạy trên CPU với tokenizer offline: `|eligible| = 49.885`; mask
  random không chứa token đặc biệt; Jaccard(rule, random) nằm trong khoảng
  0,84–0,86; tỉ lệ vị trí được tha giống hệt nhau giữa hai mask; penalty bằng 0
  khi toàn bộ xác suất dồn vào token đúng.

**Hệ quả:** các giá trị λ ∈ {0,1; 0,5} vẫn giữ nguyên, nhưng penalty mới nhỏ
hơn penalty cũ ở cùng λ. Không có số liệu cũ nào cần so, nên không mất gì.

### B3. `eval_loss` đang chứa penalty (A2)

`Trainer.prediction_step` gọi `compute_loss`, nên với λ>0 thì
`eval_loss = CE + λ·penalty`. `metric_for_best_model: eval_loss` và early
stopping vì thế **chọn checkpoint theo thước đo khác nhau giữa các λ**.

Đề xuất: `compute_loss` ghi CE thuần vào `outputs`. Ghi đè `prediction_step`
(hoặc `evaluate`) để log thêm `eval_ce`, và đặt `metric_for_best_model: eval_ce`
cho **mọi** lượt, kể cả λ=0.

### B4. Việc nhỏ

- `bf16: true` cần GPU hỗ trợ bf16 (Ampere trở lên). `train.py` nên kiểm tra
  `torch.cuda.is_bf16_supported()`. Nếu không hỗ trợ thì dừng và báo lỗi, không
  tự đổi kiểu dữ liệu, vì đổi sẽ làm mất tính so sánh giữa các lượt.
- `peft` chưa khóa phiên bản (`>=0.10`). DoRA và việc tự kiểm tra trọng số dùng
  chung thay đổi theo phiên bản. Nên khóa bản chính xác trong
  `requirements-gpu.lock` và ghi vào `run_info.json`.
- Trong `significance.py`, bootstrap đang dùng trung bình CER theo dòng. Số
  chính của luận văn là **CER cấp corpus**. H2 cần bản bootstrap theo corpus
  (tính lại tổng edit chia tổng ký tự trên các tài liệu được chọn lại), giống
  `oracle._clustered_bootstrap`.
- `docs/lo_trinh.md`: dòng định nghĩa `Δ_oracle` vẫn ghi "nhóm (a)", mâu thuẫn
  với dòng ngay dưới (a1 + seg_a1). Câu "`dora.yaml`/`loader.py` vẫn mang lỗi"
  đã cũ. Mục "commit" đang ghi chữ "bước 2b" thay vì mã hash.

---

## C. Cấu trúc cấu hình (`configs/`)

```
configs/
  dora.yaml               # giữ nguyên (đã bỏ lm_head)
  train.yaml              # giữ nguyên + metric_for_best_model: eval_ce
  sweep_step3.yaml        # MỚI: ma trận và giao thức, chỉ đọc sau khi chốt
  train_dryrun.yaml       # MỚI: ghi đè tối thiểu cho dry-run CPU
```

`configs/sweep_step3.yaml`:

```yaml
name: step3_lambda_sweep
train_config: configs/train.yaml
lora_config: configs/dora.yaml
seeds: [1, 2, 3]
conditions:                       # 5 điều kiện × 3 seed = 15 lượt
  - {id: lam0,        nesy_weight: 0.0, mask_mode: none}
  - {id: lam0.1_rule, nesy_weight: 0.1, mask_mode: rule}
  - {id: lam0.1_rand, nesy_weight: 0.1, mask_mode: random}
  - {id: lam0.5_rule, nesy_weight: 0.5, mask_mode: rule}
  - {id: lam0.5_rand, nesy_weight: 0.5, mask_mode: random}
order: seed_major                 # s1: 5 điều kiện → s2 → s3 (xem D2)
output_root: outputs/step3
eval:
  split_jsonl: data/splits/val.jsonl
  num_beams: 3
  max_new_tokens: 256
gates:                            # xem G
  fffd_line_rate_max: 0.01
  byte_vs_ascii_acc_gap_max: 0.15
```

Mask random dùng `mask_seed = seed`, tức mỗi seed có một mask riêng. Phương sai
giữa các seed vì vậy đã gồm cả phương sai do chọn mask. Điều này cần ghi rõ
trong luận văn.

---

## D. Điều phối 15 lượt chạy (`scripts/sweep.py`)

### D1. Giao diện

```
python scripts/sweep.py --config configs/sweep_step3.yaml --plan          # in 15 lệnh, không chạy
python scripts/sweep.py --config configs/sweep_step3.yaml --run           # chạy tuần tự (1 GPU)
python scripts/sweep.py --config ... --run --only lam0_s1                 # chạy một lượt
python scripts/sweep.py --config ... --run --dry-run                      # 15 lượt × 1 step trên CPU (F)
python scripts/sweep.py --config ... --status                             # bảng trạng thái từ manifest
```

### D2. Quy tắc

- **Run ID** có dạng `{condition_id}_s{seed}`, ví dụ `lam0.1_rule_s2`. Đây là
  tên thư mục và cũng là khóa trong manifest.
- **Idempotent:** bỏ qua lượt nào đã có file `DONE`. Lượt đang dở (có checkpoint
  nhưng chưa có `DONE`) được chạy tiếp bằng `--resume` với checkpoint mới nhất.
- **Thứ tự theo seed** (xong 5 điều kiện của seed 1 rồi mới sang seed 2): nếu
  hết thời gian GPU giữa chừng, dữ liệu vẫn cân bằng giữa các điều kiện.
- **Mỗi lượt gồm 3 pha:** `train.py` (nhận thêm `--run_id` và `--sweep_config`),
  rồi `evaluate.py` trên val (bắt buộc truyền `--test_jsonl data/splits/val.jsonl`;
  chỉ H2 mới được dùng test), rồi `classify_errors_oracle.py` (chỉ bắt buộc với
  `lam0_*`, các điều kiện khác chạy để tham khảo).
- **Commit bẩn thì từ chối chạy** (`git status --porcelain` không rỗng), trừ
  khi có `--allow-dirty`. Nhờ vậy mọi lượt đều gắn với một commit sạch.
- **Lỗi ở một lượt** thì ghi `status=failed` kèm đoạn log cuối, rồi chạy tiếp
  lượt sau. Không dừng cả sweep.

### D3. Manifest (`outputs/step3/manifest.csv`, có commit vào git)

`run_id, condition, seed, nesy_weight, mask_mode, git_commit, config_sha256, data_sha256,
status, started, wall_min, best_step, best_eval_ce, val_corpus_cer, val_fffd_lines,
byte_acc, ascii_acc, delta_oracle (chỉ lam0), gpu_name, peft, transformers, torch`

---

## E. Checkpoint và log

### E1. Cấu trúc thư mục

```
outputs/step3/<run_id>/
  run_info.json            # args, git commit, cấu hình đã resolve, sha256 dữ liệu, version, GPU
  train.log                # log đầy đủ
  metrics.jsonl            # mỗi lần log/eval: step, ce, penalty, eval_ce, byte_acc, ascii_acc
  checkpoint-*/            # HF Trainer, save_total_limit=2 (KHÔNG commit)
  final_adapter/           # adapter tốt nhất theo eval_ce (KHÔNG commit, sao lưu ngoài)
  eval_val.jsonl           # dự đoán val (commit được: khoảng 50 KB)
  eval_val_summary.json
  oracle/                  # summary.json, report.md, lines.jsonl (chỉ lam0 bắt buộc)
  DONE                     # chỉ ghi khi cả 3 pha thành công
```

### E2. Được commit và không được commit

- **Commit:** `manifest.csv`, `run_info.json`, `metrics.jsonl`,
  `eval_val*.json(l)`, `oracle/*`. Tất cả đều nhỏ, và đủ để tái tạo mọi bảng số
  của luận văn mà không cần GPU.
- **Không commit:** `checkpoint-*` và `final_adapter/`. Thêm vào `.gitignore`,
  rồi sao lưu `final_adapter` ra ngoài repo (Drive hoặc HF Hub private), ghi
  đường dẫn vào manifest.

### E3. Callback đo mảnh byte (A7, phục vụ ngưỡng C3.2)

Mỗi lần eval, dùng teacher forcing trên val và tính độ chính xác top-1 theo 2
nhóm token đúng:

- `byte_acc`: token đúng là mảnh byte, tức `decode([t])` chứa U+FFFD (352 id).
- `ascii_acc`: token đúng decode ra ký tự chỉ gồm ASCII.

Ghi cả hai vào `metrics.jsonl`. Không cần sinh chuỗi, và chỉ tốn thêm một phép
argmax trên logits đã có sẵn khi eval.

---

## F. Dry-run trên CPU (máy local, không GPU)

### F1. Mục tiêu

Chứng minh cả pipeline chạy thông suốt (cấu hình, dữ liệu, mask, loss, train
một step, eval, lưu, nạp lại, đánh giá, oracle) cho cả 15 lượt, trong vài phút,
**không cần GPU**.

### F2. Model cho dry-run

Hai phương án, Antigravity chọn:

| Phương án | Cách làm | Ưu | Nhược |
|---|---|---|---|
| **F-tiny (đề xuất)** | Commit `config.json`, `preprocessor_config.json`, `generation_config.json` của Florence-2 vào `data/florence2_config/` (vài KB, giống cách đã làm với tokenizer). Dựng `Florence2ForConditionalGeneration(config)` với text tower thu nhỏ (`d_model=64`, 1 lớp encoder, 1 lớp decoder) và vision tower nhỏ nhất mà config cho phép, trọng số khởi tạo ngẫu nhiên | Không tải weight, chạy vài giây, đúng tinh thần HANDOVER | Phải kiểm tra config DaViT có cho thu nhỏ không. Tên module phải giữ nguyên để `_assert_target_modules_exist` còn ý nghĩa |
| F-real | Nạp Florence-2-base thật (0,23B, khoảng 0,9 GB) trên CPU, `max_steps=1`, 4 mẫu | Kiểm đúng model thật | Phải tải weight, mỗi lượt vài phút, trái với quy tắc hiện tại của HANDOVER |

### F3. `configs/train_dryrun.yaml`

Ghi đè: `dataset_path: data/splits/train.jsonl` (không cần augment),
`max_steps: 1`, `eval_steps: 1`, `save_steps: 1`, `per_device_train_batch_size: 2`,
`gradient_accumulation_steps: 1`, `bf16: false`, `max_train_samples: 4`,
`max_eval_samples: 4`, `max_label_length: 64`. Hai khóa `max_*_samples` cần thêm
vào `train.py`.

### F4. Các kiểm tra mà dry-run phải khẳng định (assert, không chỉ in ra)

1. `sweep.py --plan` in ra đúng 15 run ID, không trùng, đúng thứ tự seed-major.
2. Với mỗi điều kiện: loss hữu hạn. Nếu λ=0 thì penalty bằng 0. Nếu λ>0 thì
   penalty > 0 và `eval_ce` được log.
3. Mask: rule có 45.775 token; random có 45.775 token, không chứa id đặc biệt,
   và khác nhau giữa các seed. Vị trí tha thứ **giống hệt nhau** giữa rule và
   random (theo B2).
4. `print_trainable_parameters()` được ghi vào `run_info.json`. Đây cũng là con
   số tỉ lệ adapter còn thiếu ở mục C2.
5. **Test chống tái phát lỗi v1:** nạp lại `final_adapter` hai lần (`merge=True`
   và `merge=False`). Logits trên cùng một batch phải khớp nhau (`atol=1e-4`),
   và embedding đầu vào trước và sau merge phải giống hệt nhau.
6. `evaluate.py` chạy trên 2 ảnh val và ghi được jsonl. Sau đó
   `classify_errors_oracle.py` chạy được trên file đó.
7. Manifest có đủ 15 dòng `status=done`. Chạy lại `--run` lần hai thì bỏ qua
   cả 15 (kiểm tra tính idempotent).
8. Callback E3 ghi được `byte_acc` và `ascii_acc` (giá trị nào cũng được, miễn
   là số hữu hạn hoặc NaN có ghi rõ lý do).

---

## G. Quy trình trên GPU và các cổng kiểm soát

| Cổng | Việc | Điều kiện qua | Không qua thì |
|---|---|---|---|
| G0 | `scripts/debug_fffd.py model` trên checkpoint v1 (khoảng 5 phút). `lo_trinh.md` ghi là bắt buộc trước bước 3 | Có ghi kết quả vào `docs/debug_fffd.md` | Vẫn chạy tiếp, nhưng giả thuyết lỗi merge chưa được xác nhận trong luận văn |
| G1 | Sinh `train_augmented.jsonl` (seed cố định), ghi sha256. Chạy dry-run F trên GPU với model thật, `max_steps=5` | Mọi assert F4 đều đạt | Dừng lại, sửa lỗi |
| G2 | Chạy trọn `lam0_s1`, đo thời gian | Val có dòng chứa U+FFFD ≤ 1 %, **và** `ascii_acc − byte_acc` ≤ 15 điểm % (C3.2) | **Dừng.** Họp để quyết phương án dự phòng (`trainable_token_indices`) trước khi chạy 14 lượt còn lại |
| G3 | 14 lượt còn lại, thứ tự seed-major | — | Lượt nào lỗi thì chạy lại riêng lượt đó |
| G4 | Bước 3b: Δ_oracle chính thức từ 3 lượt `lam0_*` trên val (A4). Áp quy tắc rẽ nhánh | — | — |
| G5 | H2: chạy test **một lần** | — | — |

Ước lượng thời gian GPU chỉ nên đưa ra sau khi xong G2. Nhân thời gian của
`lam0_s1` với khoảng 15, cộng khoảng 10 % cho các lượt λ>0 (thêm bước tha thứ).

---

## H. Giao thức thống kê (đăng ký trước, bổ sung cho `lo_trinh.md`)

**H1. Trên val.**

- Chọn λ* ∈ {0,1; 0,5} là giá trị có CER corpus trung bình 3 seed thấp hơn
  của `rule`.
- Δ_oracle lấy trung bình trên 3 seed của `lam0`. Khoảng tin cậy 95 % tính như
  sau: mỗi lần bootstrap chọn lại tài liệu, tính Δ cho từng seed rồi lấy trung
  bình 3 seed (A4).

**H2. Trên test (chạy một lần).**

- Hai so sánh, hiệu chỉnh Holm:
  1. `rule(λ*)` với `lam0`.
  2. `rule(λ*)` với `random(λ*)`.
- Với mỗi điều kiện, lấy trung bình dự đoán của 3 seed ở mức số edit và số ký
  tự của từng dòng. Chạy paired clustered bootstrap theo tài liệu (33 cụm) trên
  **CER corpus**, một phía, 10.000 lần.

**H3. Diễn giải đăng ký trước.**

- rule thắng `lam0` nhưng **không** thắng random: NeSy chỉ đóng vai một bộ điều
  chuẩn chung, không phải "tri thức âm tiết". Kết luận này vẫn công bố được.
  B1.3 cho thấy đây là khả năng đáng kể.
- rule thắng cả hai: giữ NeSy trong phần giải pháp (quy tắc hiện hành).

**H4. Bổ sung cho vùng chưa định nghĩa (A3).** Nếu Δ_oracle < 2,0 nhưng cận
trên của khoảng tin cậy ≥ 2,0, thì kết luận là **không đủ bằng chứng rằng trần
thấp**. Khi đó đi theo nhánh "Δ ≥ 2,0", tức đo phần FSM lấy được ở bước 4,
và ghi rõ khoảng tin cậy khi báo cáo.

Chưa có số liệu Bước 3 nào tồn tại, nên việc bổ sung này không vi phạm tinh
thần đăng ký trước. Tuy vậy cần ghi ngày sửa và mã commit vào `lo_trinh.md`.
