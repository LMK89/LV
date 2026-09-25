import logging

import torch
import torch.nn.functional as F
from transformers import Trainer

from vocr.nesy.syllables import is_valid_syllable, context_forgiven_rows

logger = logging.getLogger(__name__)


def compute_eligible_token_ids(tokenizer) -> set[int]:
    """
    Xác định tập các token hợp lệ để tham gia vào NeSy mask:
    - decode riêng lẻ ra chuỗi không rỗng, không toàn khoảng trắng
    - KHÔNG chứa U+FFFD (loại trừ các mảnh byte dở dang để bảo vệ byte mở đầu tiếng Việt)
    - không phải token đặc biệt (loại trừ </s>, <s>, <pad>, <unk>, v.v.)
    """
    vocab_size = len(tokenizer)
    eligible = set()
    for tid in range(vocab_size):
        try:
            decoded = tokenizer.decode([tid], skip_special_tokens=True).strip()
            if decoded and not decoded.isspace() and "\ufffd" not in decoded:
                eligible.add(tid)
        except Exception:
            pass
    return eligible


def compute_invalid_token_ids(tokenizer) -> set[int]:
    """
    Xác định tập ID các token vi phạm quy tắc âm tiết tiếng Việt trong tập eligible:
    - decode riêng lẻ không thỏa mãn is_valid_syllable(decoded)
    """
    eligible = compute_eligible_token_ids(tokenizer)
    invalid_ids = set()
    for tid in eligible:
        try:
            decoded = tokenizer.decode([tid], skip_special_tokens=True).strip()
            if not is_valid_syllable(decoded):
                invalid_ids.add(tid)
        except Exception:
            pass
    return invalid_ids


def compute_byte_and_ascii_token_ids(tokenizer) -> tuple[set[int], set[int]]:
    """
    Phân loại token thành:
    - byte_ids: token là mảnh byte UTF-8 dở dang (decode ra chuỗi chứa U+FFFD, 352 token).
    - ascii_ids: token decode ra chuỗi hoàn toàn là ký tự ASCII (ord < 128).
    Dùng để đo byte_acc vs ascii_acc khi đánh giá (cổng kiểm soát G2).
    """
    byte_ids = set()
    ascii_ids = set()
    vocab_size = len(tokenizer)
    for tid in range(vocab_size):
        try:
            dec = tokenizer.decode([tid], skip_special_tokens=True)
            if not dec:
                continue
            if "\ufffd" in dec:
                byte_ids.add(tid)
            elif all(ord(c) < 128 for c in dec):
                ascii_ids.add(tid)
        except Exception:
            pass
    return byte_ids, ascii_ids


def build_invalid_mask(tokenizer, mask_mode: str = "rule", mask_seed: int = 0) -> tuple[torch.Tensor, set[int], set[int]]:
    """
    Tạo tensor mask float32 [vocab_size], set token ID bị phạt, và set rule_invalid_ids.
    - mask_mode="rule": phạt đúng các token vi phạm âm tiết (45.775 token).
    - mask_mode="random": đối chứng công bằng (control) — lấy mẫu ngẫu nhiên đúng số lượng
      |rule| token từ tập eligible_token_ids (49.885 token). Đảm bảo không phạt token đặc biệt
      và không phạt mảnh byte lẻ.
    Trả về: (invalid_mask, active_invalid_ids, rule_invalid_ids).
    """
    vocab_size = len(tokenizer)
    rule_invalid_ids = compute_invalid_token_ids(tokenizer)
    invalid_mask = torch.zeros(vocab_size, dtype=torch.float32)

    if mask_mode == "random":
        eligible = sorted(list(compute_eligible_token_ids(tokenizer)))
        n_invalid = len(rule_invalid_ids)
        gen = torch.Generator().manual_seed(mask_seed)
        perm = torch.randperm(len(eligible), generator=gen)[:n_invalid]
        chosen = [eligible[i] for i in perm.tolist()]
        for tid in chosen:
            invalid_mask[tid] = 1.0
        return invalid_mask, set(chosen), rule_invalid_ids
    else:
        for tid in rule_invalid_ids:
            invalid_mask[tid] = 1.0
        return invalid_mask, rule_invalid_ids, rule_invalid_ids


class NeSyTrainer(Trainer):
    """
    Trainer cộng thêm NeSy penalty khả vi vào CE loss:

        L = L_CE + λ · Σ_t m_t · s_t / Σ_t m_t,   s_t = Σ_{v ∈ INVALID, v ≠ y_t} p_t(v)

    - INVALID: token mà decode riêng lẻ không phải âm tiết hợp lệ (mask_mode="rule"),
      hoặc tập ngẫu nhiên đối chứng trong eligible tokens (mask_mode="random").
    - m_t = 1 tại vị trí có nhãn và KHÔNG được cửa sổ tha thứ subword bỏ qua.
    - Không bao giờ phạt token đúng của nhãn: s_t loại trừ xác suất tại y_t để tránh
      chống lại CE khi nhãn là tên riêng / từ ngoại lai.
    - Vị trí tha thứ (context-forgiven) LUÔN tính bằng mask rule cho cả 2 điều kiện.
    - eval_ce được log riêng biệt để làm tiêu chí early stopping / chọn checkpoint công bằng.
    - Đo độ chính xác byte_acc vs ascii_acc phục vụ cổng kiểm soát G2.
    """

    def __init__(self, *args, nesy_weight: float = 0.1, tokenizer=None,
                 forgive_window: int = 5, mask_mode: str = "rule", mask_seed: int = 0,
                 **kwargs):
        super().__init__(*args, **kwargs)
        if nesy_weight < 0:
            raise ValueError("nesy_weight must be >= 0 (negative would encourage invalid syllables)")
        if mask_mode not in ("rule", "random", "none"):
            raise ValueError(f"mask_mode must be 'rule', 'random', or 'none', got {mask_mode!r}")
        self.nesy_weight = nesy_weight
        self.mask_mode = mask_mode
        self.nesy_tokenizer = tokenizer
        self.forgive_window = forgive_window

        self._pair_forgive_cache: dict[tuple[int, ...], bool] = {}
        self._invalid_token_ids: set[int] = set()
        self._rule_invalid_ids: set[int] = set()
        self._eval_ce_losses: list[float] = []

        # Phân loại phục vụ đo byte_acc và ascii_acc (cổng G2)
        self._byte_token_ids = set()
        self._ascii_token_ids = set()
        if self.nesy_tokenizer is not None:
            self._byte_token_ids, self._ascii_token_ids = compute_byte_and_ascii_token_ids(self.nesy_tokenizer)
        self._eval_byte_correct = 0
        self._eval_byte_total = 0
        self._eval_ascii_correct = 0
        self._eval_ascii_total = 0

        self.invalid_mask = None
        if self.nesy_tokenizer is not None and self.nesy_weight > 0 and self.mask_mode != "none":
            vocab_size = len(self.nesy_tokenizer)
            logger.info(f"Precomputing NeSy invalid tokens mask for vocab size {vocab_size} ({mask_mode})...")
            self.invalid_mask, self._invalid_token_ids, self._rule_invalid_ids = build_invalid_mask(
                self.nesy_tokenizer, mask_mode=self.mask_mode, mask_seed=mask_seed
            )
            logger.info(f"NeSy precomputation complete ({self.mask_mode} mask). "
                        f"Penalised tokens: {len(self._invalid_token_ids)}/{vocab_size}")

        logger.info(f"NeSyTrainer initialized | nesy_weight={nesy_weight} | mask_mode={mask_mode}")

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        outputs = model(**inputs)
        logits, labels = outputs.logits, inputs["labels"]
        ce_loss = outputs.loss

        if self.nesy_weight <= 0 or self.nesy_tokenizer is None or self.invalid_mask is None:
            return (ce_loss, outputs) if return_outputs else ce_loss

        if self.invalid_mask.device != logits.device:
            self.invalid_mask = self.invalid_mask.to(logits.device)

        batch_size, seq_len, vocab_size = logits.shape
        if self.invalid_mask.shape[-1] != vocab_size:
            if self.invalid_mask.shape[-1] < vocab_size:
                pad_len = vocab_size - self.invalid_mask.shape[-1]
                pad_tensor = torch.zeros(pad_len, dtype=self.invalid_mask.dtype, device=self.invalid_mask.device)
                self.invalid_mask = torch.cat([self.invalid_mask, pad_tensor], dim=-1)
            else:
                self.invalid_mask = self.invalid_mask[:vocab_size]

        probs = F.softmax(logits, dim=-1)

        label_mask = (labels != -100).float()  # [batch_size, seq_len]

        # ── Context-window subword forgiveness (B1) ────────────────────
        # BPE cắt âm tiết như "nghiêng" thành ["ngh","iê","ng"] — từng mảnh
        # đều fail is_valid_syllable(). Để không phạt oan, ta soi các dải token
        # liên tiếp dài 2..forgive_window: nếu decode chung ra toàn âm tiết hợp
        # lệ thì tha cả dải. Một .tolist() mỗi batch + cache quyết định decode
        # (các dải giống nhau lặp lại liên tục) để tránh sync GPU->CPU per-vị-trí.
        labels_list = labels.detach().cpu().tolist()
        check_ids = self._rule_invalid_ids if self._rule_invalid_ids else self._invalid_token_ids
        forgiven_rows = context_forgiven_rows(
            labels_list, seq_len, check_ids,
            self.nesy_tokenizer, self._pair_forgive_cache,
            max_window=self.forgive_window,
        )
        context_forgiven = torch.tensor(
            forgiven_rows, dtype=label_mask.dtype, device=label_mask.device
        )
        # ──────────────────────────────────────────────────────────────

        invalid_probs = probs * self.invalid_mask.view(1, 1, -1)
        sum_invalid_probs = invalid_probs.sum(dim=-1)  # [batch_size, seq_len]

        # Không bao giờ phạt token đúng của nhãn (loại trừ gold token khỏi penalty)
        valid_label_mask = (labels >= 0) & (labels < vocab_size)
        safe_labels = labels.clamp(min=0, max=vocab_size - 1)
        gold_probs = probs.gather(dim=-1, index=safe_labels.unsqueeze(-1)).squeeze(-1)
        gold_in_invalid = self.invalid_mask[safe_labels]
        gold_penalty = gold_probs * gold_in_invalid * valid_label_mask.float()
        sum_invalid_probs = (sum_invalid_probs - gold_penalty).clamp(min=0.0)

        # Penalise only positions that are (a) active labels AND
        # (b) NOT context-forgiven as subword fragments
        effective_mask = label_mask * (1.0 - context_forgiven)

        nesy_penalty = (
            (sum_invalid_probs * effective_mask).sum()
            / (effective_mask.sum() + 1e-8)
            * self.nesy_weight
        )
        loss = ce_loss + nesy_penalty

        if self.state.global_step % 50 == 0:
            bad_count     = (sum_invalid_probs * effective_mask > 0.5).sum().item()
            forgiven_count = int(context_forgiven.sum().item())
            logger.info(
                f"Step {self.state.global_step}: CE={ce_loss.item():.4f} "
                f"NeSyPenalty={nesy_penalty.item():.4f} "
                f"(bad={bad_count}, subword-forgiven={forgiven_count})"
            )

        return (loss, outputs) if return_outputs else loss

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        loss, logits, labels = super().prediction_step(
            model, inputs, prediction_loss_only=prediction_loss_only, ignore_keys=ignore_keys
        )
        if hasattr(self, "_eval_ce_losses") and isinstance(self._eval_ce_losses, list):
            with torch.no_grad():
                outputs = model(**inputs)
                if outputs.loss is not None:
                    self._eval_ce_losses.append(outputs.loss.detach().mean().item())

        # Đo byte_acc và ascii_acc (cổng G2)
        if logits is not None and labels is not None and getattr(self, "_byte_token_ids", None):
            with torch.no_grad():
                preds = logits.argmax(dim=-1)
                valid_mask = (labels >= 0) & (labels != -100)
                labels_flat = labels[valid_mask].view(-1)
                preds_flat = preds[valid_mask].view(-1)
                if len(labels_flat) > 0:
                    labels_list = labels_flat.tolist()
                    preds_list = preds_flat.tolist()
                    for y, p in zip(labels_list, preds_list):
                        if y in self._byte_token_ids:
                            self._eval_byte_total += 1
                            if y == p:
                                self._eval_byte_correct += 1
                        elif y in self._ascii_token_ids:
                            self._eval_ascii_total += 1
                            if y == p:
                                self._eval_ascii_correct += 1

        return loss, logits, labels

    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
        self._eval_ce_losses = []
        self._eval_byte_correct = 0
        self._eval_byte_total = 0
        self._eval_ascii_correct = 0
        self._eval_ascii_total = 0

        metrics = super().evaluate(eval_dataset=eval_dataset, ignore_keys=ignore_keys, metric_key_prefix=metric_key_prefix)

        if self._eval_ce_losses:
            metrics[f"{metric_key_prefix}_ce"] = sum(self._eval_ce_losses) / len(self._eval_ce_losses)
        else:
            metrics[f"{metric_key_prefix}_ce"] = metrics.get(f"{metric_key_prefix}_loss", 0.0)

        # Ghi byte_acc và ascii_acc (phục vụ ngưỡng báo động C3.2 và cổng G2)
        byte_acc = (self._eval_byte_correct / self._eval_byte_total) if self._eval_byte_total > 0 else 0.0
        ascii_acc = (self._eval_ascii_correct / self._eval_ascii_total) if self._eval_ascii_total > 0 else 0.0
        metrics[f"{metric_key_prefix}_byte_acc"] = round(byte_acc, 4)
        metrics[f"{metric_key_prefix}_ascii_acc"] = round(ascii_acc, 4)
        metrics[f"{metric_key_prefix}_acc_gap"] = round(ascii_acc - byte_acc, 4)

        return metrics
