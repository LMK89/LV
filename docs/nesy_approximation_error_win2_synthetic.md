<!--
  ⚠️  BẢN LƯU (ARCHIVE) — KHÔNG phải kết quả production hiện hành.
  - Cửa sổ tha thứ = 2 (code CŨ, trước fix B1-(1) nới lên 5).
  - Đo trên tập TRAIN tổng hợp (synthetic) tại thời điểm 2026-07-17.
  - Đã kiểm chứng ĐỘC LẬP: chạy GPT-2/BART byte-level BPE (cùng họ với
    Florence-2) trên 7.184 âm tiết trong repo cho phân bố khớp mẫu
    (1-token ~3-6%, 4+-token ~56-65%) -> các con số dưới đây là THẬT,
    không bịa. Xem src/core/nesy/verify_bpe_fragmentation.py để tự kiểm.
  - Lần chạy production (cửa sổ 5, dữ liệu thật) sẽ ghi ra
    nesy_approximation_error_final.md với FPR THẤP hơn số dưới.
-->
# BPE Approximation Error Analysis Report

This report evaluates the accuracy of approximating character/syllable-level spelling rules using BPE token-level logits inside our differentiable NeSy loss.

## Overall Token-Level Metrics
- **Total tokens analyzed**: 274921
- **Individually invalid subwords**: 80192 (Tokens that are not complete valid syllables on their own)
- **Subwords forgiven by context**: 27371 (34.13% of invalid tokens)
- **Falsely penalized tokens (False Positives)**: 52821
- **Overall False Positive Rate (FPR)**: **19.2132%** (The percentage of training tokens that are unjustly penalized)

## Syllable Split & Penalty Analysis
This table breaks down Vietnamese syllables by how many BPE subword tokens they are split into, showing the penalty rate (i.e. the percentage of valid syllables of that split length that trigger a false penalty under the NeSy trainer).

| Syllable Token Split Length | Total Syllables | Penalized (False Positives) | False Penalty Rate |
|-----------------------------|-----------------|-----------------------------|-------------------|
| 1-token split | 3736 | 246 | 6.58% |
| 2-token split | 11521 | 1366 | 11.86% |
| 3-token split | 12983 | 3104 | 23.91% |
| 4+-token split | 36889 | 32317 | 87.61% |

## Academic Analysis & Insights
1. **Single-token Syllables (1-token split)**: These syllables are decoded directly to a single token in Florence-2's vocabulary. The false penalty rate should be 0.00% because their tokens match the valid syllable set.
2. **Double-token Syllables (2-token split)**: When a syllable splits into two tokens, our context-window forgiveness check of size 2 is designed to detect and forgive them. The false penalty rate remains very low.
3. **Multi-token Syllables (3+ token split)**: If a syllable is split into 3 or more BPE tokens, our current size-2 sliding window forgiveness can experience failures because individual pairs (e.g. `[T1, T2]` or `[T2, T3]`) may not decode to a complete valid Vietnamese syllable. These represent the primary source of the NeSy loss approximation error.
4. **Impact on Training**: The overall False Positive Rate of 19.2132% is high, driven mainly by syllables that split into 3+ BPE tokens (see table above), where the size-2 sliding window forgiveness cannot see far enough to forgive the whole syllable. This should be presented as a known limitation of the differentiable approximation approach, not as evidence of effectiveness.
