import logging

import torch
import torch.nn.functional as F
from transformers import Trainer

from vocr.nesy.syllables import is_valid_syllable, context_forgiven_rows

logger = logging.getLogger(__name__)


class NeSyTrainer(Trainer):
    """
    Trainer cộng thêm NeSy penalty khả vi vào CE loss:

        L = L_CE + λ · Σ_t m_t · s_t / Σ_t m_t,   s_t = Σ_{v ∈ INVALID} p_t(v)

    - INVALID: token mà decode riêng lẻ không phải âm tiết hợp lệ (mask_mode="rule"),
      hoặc một tập ngẫu nhiên cùng kích thước (mask_mode="random", control).
    - m_t = 1 tại vị trí có nhãn và KHÔNG được cửa sổ tha thứ subword bỏ qua.
    - Gradient đi qua softmax -> logits -> adapter.

    Lưu ý khi phân tích: dưới teacher forcing, CE đã đẩy khối xác suất khỏi mọi
    token ≠ nhãn, nên penalty chủ yếu (a) trùng hướng với CE khi nhãn hợp lệ,
    (b) ngược hướng CE khi nhãn là mảnh subword bị phạt oan.
    """

    def __init__(self, *args, nesy_weight: float = 0.1, tokenizer=None,
                 forgive_window: int = 5, mask_mode: str = "rule", mask_seed: int = 0,
                 **kwargs):
        super().__init__(*args, **kwargs)
        if nesy_weight < 0:
            raise ValueError("nesy_weight must be >= 0 (negative would encourage invalid syllables)")
        if mask_mode not in ("rule", "random"):
            raise ValueError(f"mask_mode must be 'rule' or 'random', got {mask_mode!r}")
        self.nesy_weight = nesy_weight
        self.mask_mode = mask_mode
        self.nesy_tokenizer = tokenizer
        # Cửa sổ tha thứ subword (B1): 2 chỉ cứu âm tiết tách 2 token; nâng lên
        # ~5 để cứu cả âm tiết bị BPE cắt 3-4 mảnh (giảm phạt oan / FPR).
        self.forgive_window = forgive_window

        # Token-window -> "forgiven" decisions are expensive (tokenizer.decode +
        # regex validation) but the same BPE windows recur constantly across
        # steps/epochs, so we cache them for the lifetime of training instead
        # of recomputing on every forward pass.
        self._pair_forgive_cache: dict[tuple[int, ...], bool] = {}
        self._invalid_token_ids: set[int] = set()

        self.invalid_mask = None
        if self.nesy_tokenizer is not None and self.nesy_weight > 0:
            vocab_size = len(self.nesy_tokenizer)
            logger.info(f"Precomputing NeSy invalid tokens mask for vocab size {vocab_size}...")
            invalid_mask_cpu = torch.zeros(vocab_size, dtype=torch.float32)
            for tid in range(vocab_size):
                try:
                    decoded = self.nesy_tokenizer.decode([tid], skip_special_tokens=True).strip()
                    if decoded and not decoded.isspace() and not is_valid_syllable(decoded):
                        invalid_mask_cpu[tid] = 1.0
                        self._invalid_token_ids.add(tid)
                except Exception:
                    pass
            if mask_mode == "random":
                # Control: giữ nguyên SỐ token bị phạt nhưng chọn ngẫu nhiên, để
                # tách tác dụng tri thức âm tiết khỏi tác dụng điều chuẩn chung.
                n_invalid = int(invalid_mask_cpu.sum())
                gen = torch.Generator().manual_seed(mask_seed)
                chosen = torch.randperm(vocab_size, generator=gen)[:n_invalid]
                invalid_mask_cpu = torch.zeros(vocab_size, dtype=torch.float32)
                invalid_mask_cpu[chosen] = 1.0
                self._invalid_token_ids = set(chosen.tolist())
            self.invalid_mask = invalid_mask_cpu
            logger.info(f"NeSy precomputation complete ({mask_mode} mask). "
                        f"Penalised tokens: {int(invalid_mask_cpu.sum())}/{vocab_size}")

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
        forgiven_rows = context_forgiven_rows(
            labels_list, seq_len, self._invalid_token_ids,
            self.nesy_tokenizer, self._pair_forgive_cache,
            max_window=self.forgive_window,
        )
        context_forgiven = torch.tensor(
            forgiven_rows, dtype=label_mask.dtype, device=label_mask.device
        )
        # ──────────────────────────────────────────────────────────────

        invalid_probs = probs * self.invalid_mask.view(1, 1, -1)
        sum_invalid_probs = invalid_probs.sum(dim=-1)  # [batch_size, seq_len]

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
