"""Ràng buộc âm tiết lúc suy luận (LogitsProcessor)."""

try:
    import torch
except ImportError:
    torch = None

try:
    from transformers import LogitsProcessor
except ImportError:
    class LogitsProcessor:
        pass

from vocr.nesy.syllables import could_start_valid_syllable, is_valid_syllable


class VietnamesePhonologyLogitsProcessor(LogitsProcessor):
    """Ràng buộc âm vị học tiếng Việt lúc giải mã (inference-time).

    Hai chế độ (`mode`):
      - **"automaton"** (mặc định, B3-2): automaton âm tiết CẤP KÝ TỰ CÓ TRẠNG
        THÁI. Theo dõi âm tiết đang được dựng (các token kể từ ranh giới từ gần
        nhất) và chỉ cho phép một ứng viên nếu nó giữ âm tiết hiện tại là TIỀN
        TỐ của một âm tiết hợp lệ, hoặc đóng một âm tiết ĐÃ hợp lệ để mở từ mới.
        Nhờ vậy âm tiết bị BPE cắt nhiều token vẫn dựng được (không bị chặn giữa
        chừng) mà vẫn loại chuỗi phi cấu trúc. Có VAN AN TOÀN: nếu mọi ứng viên
        top-k đều bị chặn thì KHÔNG ràng buộc bước đó (tránh deadlock / lặp vô
        hạn — hiện tượng "self-intoxication").
      - **"token"** (B3-1): kiểm từng token rời rạc, nhẹ hơn — dùng để đối chiếu
        (ablation B3-1 vs B3-2).

    Với encoder-decoder (Florence-2), `input_ids` mà processor nhận là chuỗi
    DECODER đã sinh nên không lẫn prompt. Ký tự nhiều byte bị BPE cắt dở được
    phát hiện qua ký tự thay thế U+FFFD khi decode -> cho phép (đang dựng ký tự).
    """

    def __init__(
        self,
        tokenizer,
        top_k: int = 50,
        penalty: float = -1e4,
        skip_special_tokens: bool = True,
        warmup_tokens: int = 1,
        mode: str = "automaton",
        max_scan_tokens: int = 16,
    ):
        self.tokenizer = tokenizer
        self.top_k = top_k
        self.penalty = penalty
        self.skip_special_tokens = skip_special_tokens
        self.warmup_tokens = warmup_tokens
        self.mode = mode
        self.max_scan_tokens = max_scan_tokens
        self._cache: dict[int, bool] = {}            # cache cho mode "token"
        self._wordstart_cache: dict[int, bool] = {}
        self._special_ids = {
            i for i in (
                getattr(tokenizer, "eos_token_id", None),
                getattr(tokenizer, "bos_token_id", None),
                getattr(tokenizer, "pad_token_id", None),
                getattr(tokenizer, "unk_token_id", None),
                getattr(tokenizer, "decoder_start_token_id", None),
            ) if i is not None
        }

    def _is_word_start(self, token_id: int) -> bool:
        """Token có MỞ ĐẦU một từ mới không (byte-level BPE đánh dấu 'Ġ',
        SentencePiece dùng '▁'); token không có dấu này là mảnh nối GIỮA từ."""
        cached = self._wordstart_cache.get(token_id)
        if cached is not None:
            return cached
        try:
            tok = self.tokenizer.convert_ids_to_tokens(token_id)
            res = bool(tok) and (tok.startswith("Ġ") or tok.startswith("▁"))
        except Exception:
            res = True
        self._wordstart_cache[token_id] = res
        return res

    def _decode(self, ids) -> str:
        try:
            return self.tokenizer.decode(ids, skip_special_tokens=True)
        except Exception:
            return ""

    # ── Automaton âm tiết (B3-2) ──────────────────────────────────────
    def _current_word_ids(self, row):
        """Token của âm tiết ĐANG dựng: từ ranh giới từ gần nhất tới cuối chuỗi."""
        ids = []
        for tid in reversed(row):
            if tid in self._special_ids:
                break
            ids.append(tid)
            if self._is_word_start(tid) or len(ids) >= self.max_scan_tokens:
                break
        ids.reverse()
        return ids

    @staticmethod
    def _complete_ok(text: str) -> bool:
        """Âm tiết vừa hoàn tất có hợp lệ không (để cho phép đóng từ / kết câu)."""
        text = text.strip()
        if not text:
            return True
        if "�" in text:   # còn dở dang ký tự nhiều byte -> chưa nên đóng
            return False
        return all(is_valid_syllable(w) for w in text.split())

    @staticmethod
    def _prefix_ok(text: str) -> bool:
        """Âm tiết đang dựng (đã nối ứng viên) còn là tiền tố hợp lệ không."""
        text = text.strip()
        if not text:
            return True
        if "�" in text:   # đang dựng ký tự nhiều byte -> cho phép
            return True
        words = text.split()
        if not words:
            return True
        *complete, last = words
        if any(not is_valid_syllable(w) for w in complete):
            return False
        return could_start_valid_syllable(last)

    # ── Chế độ "token" (B3-1) ─────────────────────────────────────────
    def _is_token_valid(self, token_id: int) -> bool:
        if token_id in self._cache:
            return self._cache[token_id]

        if self.skip_special_tokens and token_id in (
            self.tokenizer.eos_token_id,
            self.tokenizer.bos_token_id,
            self.tokenizer.pad_token_id,
            self.tokenizer.unk_token_id,
        ):
            self._cache[token_id] = True
            return True

        try:
            decoded = self.tokenizer.decode(
                [token_id], skip_special_tokens=self.skip_special_tokens
            ).strip()
        except Exception:
            self._cache[token_id] = True
            return True

        if not decoded or decoded.isspace():
            self._cache[token_id] = True
            return True

        # B3: mảnh subword nối GIỮA từ (vd 'iê','ng' trong 'nghiêng') KHÔNG bị
        # chặn — chặn thì âm tiết nhiều token không bao giờ dựng được. Chỉ token
        # MỞ ĐẦU từ mới bị soi.
        if not self._is_word_start(token_id):
            self._cache[token_id] = True
            return True

        # Token mở đầu từ: hợp lệ nếu mỗi 'từ' là âm tiết hợp lệ HOẶC là tiền tố
        # của một âm tiết hợp lệ (cho phép mở đầu âm tiết nhiều token).
        words = decoded.split()
        result = all(could_start_valid_syllable(w) for w in words) if words else True

        self._cache[token_id] = result
        return result

    def __call__(
        self,
        input_ids: "torch.LongTensor",
        scores: "torch.FloatTensor",
    ) -> "torch.FloatTensor":
        current_len = input_ids.shape[1]

        # Fix v2: phai chan id "ma" (ngoai range tokenizer that) TRUOC CA khi kiem
        # tra warmup - vi buoc warmup (vd token dau tien, warmup_tokens=1) van goi
        # embed_tokens o buoc sau. Da xac nhan truc tiep qua safetensors: lm_head
        # that su co 51328 hang nhung tokenizer chi co 51290 token (51328-768 vs
        # 51290) - 38 hang "ma" nay khong duoc phep thang argmax o BAT KY buoc nao.
        n_tok = len(self.tokenizer)
        if scores.shape[-1] > n_tok:
            scores[:, n_tok:] = float("-inf")

        if current_len <= self.warmup_tokens:
            return scores

        batch_size, vocab_size = scores.shape
        k = min(self.top_k, vocab_size)
        top_k_ids = torch.topk(scores, k=k, dim=-1).indices

        for b in range(batch_size):
            cand = top_k_ids[b].tolist()

            # ── Chế độ "token" (B3-1): kiểm từng token rời rạc ──
            if self.mode == "token":
                for tid in cand:
                    if not self._is_token_valid(tid):
                        scores[b, tid] = self.penalty
                continue

            # ── Chế độ "automaton" (B3-2): ràng buộc theo âm tiết đang dựng ──
            cur_ids = self._current_word_ids(input_ids[b].tolist())
            # decode âm tiết hiện tại MỘT lần, dùng chung cho mọi ứng viên
            # đóng-từ (word-start/special); ứng viên nối-giữa-từ decode riêng.
            cur_complete_ok = self._complete_ok(self._decode(cur_ids))

            decisions = []
            for tid in cand:
                if tid in self._special_ids or self._is_word_start(tid):
                    ok = cur_complete_ok               # âm tiết hiện tại phải hợp lệ
                else:
                    ok = self._prefix_ok(self._decode(cur_ids + [tid]))
                decisions.append((tid, ok))

            # Van an toàn: nếu chặn HẾT top-k -> bỏ ràng buộc bước này
            if not any(ok for _, ok in decisions):
                continue
            for tid, ok in decisions:
                if not ok:
                    scores[b, tid] = self.penalty

        return scores
