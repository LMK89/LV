import argparse
import json
import os
import random
import numpy as np

def load_results(path: str) -> dict[str, dict]:
    results = {}
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Result file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    rec = json.loads(line)
                    img = rec.get("image", "")
                    if img:
                        results[img] = rec
                except Exception:
                    continue
    return results

def get_sig_level(p: float) -> str:
    if p < 0.001:
        return "*** (Very Highly Sig, p < 0.001)"
    elif p < 0.01:
        return "** (Highly Sig, p < 0.01)"
    elif p < 0.05:
        return "* (Sig, p < 0.05)"
    else:
        return "ns (Not Statistically Sig, p >= 0.05)"

def paired_bootstrap_test(file_a: str, file_b: str, n_samples: int = 1000, seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)

    res_a = load_results(file_a)
    res_b = load_results(file_b)

    # Find common images
    common_keys = sorted(list(set(res_a.keys()) & set(res_b.keys())))
    n = len(common_keys)
    if n == 0:
        print("Error: No common images found between the two results files.")
        return

    print(f"Aligning evaluation: found {n} matched predictions.")

    cer_a_all = np.array([res_a[k]["cer"] for k in common_keys])
    wer_a_all = np.array([res_a[k]["wer"] for k in common_keys])
    cer_b_all = np.array([res_b[k]["cer"] for k in common_keys])
    wer_b_all = np.array([res_b[k]["wer"] for k in common_keys])

    mean_cer_a = np.mean(cer_a_all)
    mean_wer_a = np.mean(wer_a_all)
    mean_cer_b = np.mean(cer_b_all)
    mean_wer_b = np.mean(wer_b_all)

    print(f"Overall Model A (Base/Reference): Avg CER = {mean_cer_a*100:.3f}%, Avg WER = {mean_wer_a*100:.3f}%")
    print(f"Overall Model B (Comparison):     Avg CER = {mean_cer_b*100:.3f}%, Avg WER = {mean_wer_b*100:.3f}%")

    cer_diff_obs = mean_cer_a - mean_cer_b
    wer_diff_obs = mean_wer_a - mean_wer_b
    print(f"Observed Improvement (A - B):     CER = {cer_diff_obs*100:.3f}%, WER = {wer_diff_obs*100:.3f}%")

    # Bootstrap lists
    boot_cer_a, boot_cer_b = [], []
    boot_wer_a, boot_wer_b = [], []
    boot_cer_diff, boot_wer_diff = [], []

    print(f"Running paired bootstrap resampling ({n_samples} iterations)...")
    for _ in range(n_samples):
        indices = np.random.choice(n, size=n, replace=True)
        
        # Mean CER/WER for iteration
        sub_cer_a = np.mean(cer_a_all[indices])
        sub_wer_a = np.mean(wer_a_all[indices])
        sub_cer_b = np.mean(cer_b_all[indices])
        sub_wer_b = np.mean(wer_b_all[indices])
        
        boot_cer_a.append(sub_cer_a)
        boot_wer_a.append(sub_wer_a)
        boot_cer_b.append(sub_cer_b)
        boot_wer_b.append(sub_wer_b)
        
        boot_cer_diff.append(sub_cer_a - sub_cer_b)
        boot_wer_diff.append(sub_wer_a - sub_wer_b)

    # Convert to numpy arrays
    boot_cer_a = np.array(boot_cer_a)
    boot_cer_b = np.array(boot_cer_b)
    boot_wer_a = np.array(boot_wer_a)
    boot_wer_b = np.array(boot_wer_b)
    boot_cer_diff = np.array(boot_cer_diff)
    boot_wer_diff = np.array(boot_wer_diff)

    # 95% Confidence Intervals
    ci_cer_a = (np.percentile(boot_cer_a, 2.5), np.percentile(boot_cer_a, 97.5))
    ci_wer_a = (np.percentile(boot_wer_a, 2.5), np.percentile(boot_wer_a, 97.5))
    ci_cer_b = (np.percentile(boot_cer_b, 2.5), np.percentile(boot_cer_b, 97.5))
    ci_wer_b = (np.percentile(boot_wer_b, 2.5), np.percentile(boot_wer_b, 97.5))
    ci_cer_diff = (np.percentile(boot_cer_diff, 2.5), np.percentile(boot_cer_diff, 97.5))
    ci_wer_diff = (np.percentile(boot_wer_diff, 2.5), np.percentile(boot_wer_diff, 97.5))

    # Calculate p-value (one-sided hypothesis: B is better than A, so difference A - B > 0)
    # P-value is the fraction of times the bootstrap difference is <= 0 (no improvement or worsen)
    p_cer = np.sum(boot_cer_diff <= 0) / n_samples
    p_wer = np.sum(boot_wer_diff <= 0) / n_samples

    # Formatting p-value for printing
    p_cer_str = f"{p_cer:.4f}" if p_cer > 0 else f"< {1/n_samples:.4f}"
    p_wer_str = f"{p_wer:.4f}" if p_wer > 0 else f"< {1/n_samples:.4f}"

    report = f"""# Paired Bootstrap Resampling Significance Report

This report compares two evaluation runs using non-parametric paired bootstrap resampling to measure the statistical significance of performance gains.

- **Model A (Reference)**: {os.path.basename(file_a)}
- **Model B (Comparison)**: {os.path.basename(file_b)}
- **Sample size ($N$)**: {n} paired predictions
- **Bootstrap iterations ($B$)**: {n_samples}
- **Random seed**: {seed}

## 1. Metric Performance & 95% Confidence Intervals (CI)

| Model | Metric | Mean Score | 95% Confidence Interval |
|-------|--------|------------|-------------------------|
| **Model A (Reference)** | CER | {mean_cer_a*100:.3f}% | [{ci_cer_a[0]*100:.3f}%, {ci_cer_a[1]*100:.3f}%] |
| | WER | {mean_wer_a*100:.3f}% | [{ci_wer_a[0]*100:.3f}%, {ci_wer_a[1]*100:.3f}%] |
| **Model B (Comparison)** | CER | {mean_cer_b*100:.3f}% | [{ci_cer_b[0]*100:.3f}%, {ci_cer_b[1]*100:.3f}%] |
| | WER | {mean_wer_b*100:.3f}% | [{ci_wer_b[0]*100:.3f}%, {ci_wer_b[1]*100:.3f}%] |

## 2. Statistical Significance Analysis

We test the hypothesis that **Model B improves upon Model A** (i.e. Difference = $\\text{{Score}}_A - \\text{{Score}}_B > 0$).

| Metric | Observed Difference | 95% CI of Difference | p-value | Significance Level |
|--------|---------------------|----------------------|---------|--------------------|
| **CER** | {cer_diff_obs*100:+.3f}% | [{ci_cer_diff[0]*100:.3f}%, {ci_cer_diff[1]*100:.3f}%] | {p_cer_str} | {get_sig_level(p_cer)} |
| **WER** | {wer_diff_obs*100:+.3f}% | [{ci_wer_diff[0]*100:.3f}%, {ci_wer_diff[1]*100:.3f}%] | {p_wer_str} | {get_sig_level(p_wer)} |

### Key Insights:
- A **p-value < 0.05** indicates that the improvement is statistically significant at the 95% confidence level (rejecting the null hypothesis that the difference is due to random variance in the test set).
- If the **95% CI of Difference** does not contain **0.00%**, the improvement is statistically significant.
"""
    return report


def main():
    parser = argparse.ArgumentParser(description="Paired Bootstrap Resampling Test for OCR Metrics")
    parser.add_argument("--file_a", type=str, required=True, help="Path to Model A (Baseline/Reference) detailed results .jsonl")
    parser.add_argument("--file_b", type=str, required=True, help="Path to Model B (Proposed/Improved) detailed results .jsonl")
    parser.add_argument("--n_samples", type=int, default=1000, help="Number of bootstrap iterations")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output_path", type=str, default=None, help="Save report to this path")
    args = parser.parse_args()

    try:
        report = paired_bootstrap_test(args.file_a, args.file_b, args.n_samples, args.seed)
        if report:
            print("=" * 60)
            print("SIGNIFICANCE TEST RESULTS")
            print("=" * 60)
            print(report)
            print("=" * 60)
            
            if args.output_path:
                os.makedirs(os.path.dirname(os.path.abspath(args.output_path)), exist_ok=True)
                with open(args.output_path, "w", encoding="utf-8") as f:
                    f.write(report)
                print(f"Report saved to: {args.output_path}")
    except Exception as e:
        print(f"Error executing significance test: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
