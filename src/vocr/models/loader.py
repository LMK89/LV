"""
loader.py — Nạp Florence-2 (bản NATIVE của transformers) + gắn DoRA/LoRA (peft).

Điểm nạp model DUY NHẤT của repo: train/evaluate/analysis
đều phải đi qua đây, để việc đổi backend model chỉ sửa một chỗ.

──────────────────────────────────────────────────────────────────────────────
Vì sao bỏ `trust_remote_code=True` (fix bug "Florence2LanguageConfig has no
attribute 'forced_bos_token_id'"):

Repo `microsoft/Florence-2-*` nạp bằng remote code (`configuration_florence2.py`
+ `modeling_florence2.py`) viết cho API transformers 4.4x. Từ transformers 5.x,
các tham số sinh văn bản (`forced_bos_token_id`, `forced_eos_token_id`,
`num_beams`, ...) đã bị gỡ khỏi `PretrainedConfig` và chuyển hẳn sang
`GenerationConfig`. Đoạn tương thích-ngược kế thừa từ BART trong
`Florence2LanguageConfig.__init__` vẫn đọc `self.forced_bos_token_id` -> nổ
AttributeError ngay lúc dựng config, trước cả khi tải weight. Vá tay thuộc tính
đó chỉ dời lỗi sang tầng modeling (Cache API, attention mask helper... cũng đã
đổi).

Cách sửa đúng: transformers đã có Florence-2 native
(`Florence2ForConditionalGeneration`) cùng checkpoint đã convert ở org
`florence-community`, nạp KHÔNG cần remote code -> không còn phụ thuộc vào code
4.4x, và cũng không cần vá `check_imports` để né flash_attn/einops/timm nữa.
──────────────────────────────────────────────────────────────────────────────
"""
import json
import logging
import os
from typing import Optional

import torch
from transformers import AutoProcessor

logger = logging.getLogger(__name__)

DEFAULT_BASE_MODEL_ID = "florence-community/Florence-2-base"

# Checkpoint cũ (adapter_config.json do peft ghi, config YAML, --base_model trên
# dòng lệnh) vẫn trỏ tới id `microsoft/...`. Map sang id native để mọi thứ đã
# lưu trước đây vẫn nạp được mà không phải sửa file checkpoint.
_LEGACY_ID_MAP = {
    "microsoft/Florence-2-base": "florence-community/Florence-2-base",
    "microsoft/Florence-2-base-ft": "florence-community/Florence-2-base-ft",
    "microsoft/Florence-2-large": "florence-community/Florence-2-large",
    "microsoft/Florence-2-large-ft": "florence-community/Florence-2-large-ft",
}


def resolve_base_model_id(model_id: Optional[str] = None) -> str:
    """Chuẩn hoá id model gốc: id `microsoft/...` -> bản native `florence-community/...`.

    Ưu tiên biến môi trường FLORENCE2_BASE_ID (dùng khi server chạy offline và
    phải trỏ vào thư mục model tải sẵn).
    """
    override = os.environ.get("FLORENCE2_BASE_ID")
    if override:
        return override
    if not model_id:
        return DEFAULT_BASE_MODEL_ID
    mapped = _LEGACY_ID_MAP.get(model_id)
    if mapped:
        logger.info(
            "Đổi model id %r -> %r (bản native, không cần trust_remote_code).",
            model_id, mapped,
        )
        return mapped
    return model_id


def _florence2_class():
    """Lớp model native; báo lỗi rõ ràng nếu transformers quá cũ."""
    try:
        from transformers import Florence2ForConditionalGeneration
    except ImportError as exc:  # transformers < 4.56 chưa có Florence-2 native
        import transformers
        raise ImportError(
            f"transformers=={transformers.__version__} chưa có "
            "Florence2ForConditionalGeneration (Florence-2 native). Cài đúng "
            "phiên bản đã pin: pip install -r requirements.txt"
        ) from exc
    return Florence2ForConditionalGeneration


_DTYPE_ALIASES = {
    "fp32": torch.float32, "float32": torch.float32,
    "bf16": torch.bfloat16, "bfloat16": torch.bfloat16,
    "fp16": torch.float16, "float16": torch.float16,
}


def _model_load_dtype(train_cfg: dict):
    """Dtype để NẠP weight — cố ý KHÔNG suy ra từ cờ `bf16`/`fp16`.

    `bf16: true` trong config nghĩa là "bật mixed precision", và việc đó đã do
    `TrainingArguments(bf16=True)` lo: forward/backward chạy bf16, còn master
    weight + optimizer state giữ fp32. Nếu ở ĐÂY cũng nạp weight bf16 thì thành
    PURE bf16 — AdamW giữ exp_avg/exp_avg_sq cùng dtype với param, tức ~3 chữ số
    thập phân, trong khi DoRA r=24 với lr 5e-4 toàn cộng dồn update rất nhỏ ->
    update bị nuốt, loss dễ phân kỳ. Đó chính là lý do config gốc để `fp32: true`.

    Nên mặc định LUÔN fp32 (Florence-2-base 230M ~0.9GB, thừa sức trên 16GB).
    Muốn pure bf16 thật (tiết kiệm VRAM, chấp nhận rủi ro) thì phải khai báo
    TƯỜNG MINH `model_load_dtype: bf16` trong config — không bật lén qua `bf16`.
    """
    requested = train_cfg.get("model_load_dtype")
    if requested is None:
        return torch.float32
    key = str(requested).strip().lower()
    if key not in _DTYPE_ALIASES:
        raise ValueError(
            f"model_load_dtype={requested!r} không hợp lệ. "
            f"Chọn một trong: {sorted(_DTYPE_ALIASES)}"
        )
    return _DTYPE_ALIASES[key]


def resolve_lora_config(path: str) -> dict:
    """Đọc YAML DoRA/LoRA (configs/dora.yaml) -> dict."""
    import yaml
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_processor(path_or_id: Optional[str] = None):
    """Nạp processor Florence-2 (native). Nhận cả model id lẫn thư mục checkpoint."""
    if path_or_id and os.path.isdir(path_or_id):
        return AutoProcessor.from_pretrained(path_or_id)
    return AutoProcessor.from_pretrained(resolve_base_model_id(path_or_id))


def load_base_model(model_id: Optional[str] = None, dtype=torch.float32, device: Optional[str] = None):
    """Nạp Florence-2 gốc (chưa gắn adapter). Hỗ trợ F-tiny khi DRY_RUN=1."""
    base_id = resolve_base_model_id(model_id)
    florence_cls = _florence2_class()

    if os.environ.get("DRY_RUN") == "1":
        from transformers import AutoConfig
        logger.info("Initializing F-tiny model (uninitialized weights, 1 encoder/decoder layer) for dry-run...")
        config = AutoConfig.from_pretrained(base_id)
        if hasattr(config, "text_config"):
            config.text_config.encoder_layers = 1
            config.text_config.decoder_layers = 1
        model = florence_cls(config).to(dtype=dtype)
        if device is not None:
            model = model.to(device)
        return model

    model = florence_cls.from_pretrained(
        base_id,
        # eager: bám đúng hành vi cũ của repo và né lỗi _supports_sdpa từng gặp
        # ở vài bản transformers; đổi sang "sdpa" thì phải đo lại tốc độ/CER.
        attn_implementation="eager",
        dtype=dtype,
    )
    if device is not None:
        model = model.to(device)
    return model


def _assert_target_modules_exist(model, target_modules) -> None:
    """Chặn cấu hình DoRA trỏ vào module KHÔNG tồn tại.

    peft bỏ qua trong im lặng các tên không khớp: adapter vẫn tạo được nhưng
    nhỏ hơn ý định, và không có gì báo lỗi — đúng loại hỏng ngầm làm kết quả
    luận văn sai mà không ai biết. Ở đây ta chuyển nó thành lỗi thấy được.
    """
    if not target_modules or isinstance(target_modules, str):
        return  # peft cho phép truyền regex dạng chuỗi — không kiểm được theo tên
    suffixes = {name.rsplit(".", 1)[-1] for name, _ in model.named_modules() if name}
    missing = [t for t in target_modules if t not in suffixes]
    if missing:
        raise ValueError(
            f"target_modules không khớp module nào trong model: {missing}. "
            f"Tên module hợp lệ (ví dụ): {sorted(suffixes)[:40]}. "
            "Sửa configs/dora.yaml — để nguyên thì DoRA sẽ gắn thiếu "
            "lớp mà không báo lỗi."
        )


def load_model_and_processor(lora_cfg: dict, train_cfg: dict):
    """Nạp Florence-2 base + gắn adapter DoRA theo lora_cfg.

    Returns: (model, processor, peft_config). Phần tử thứ 3 được train.py
    nhận bằng `_` (không dùng).
    """
    from peft import LoraConfig, get_peft_model

    base_model_id = resolve_base_model_id(lora_cfg.get("base_model_id"))
    dtype = _model_load_dtype(train_cfg)

    processor = AutoProcessor.from_pretrained(base_model_id)
    model = load_base_model(base_model_id, dtype=dtype)

    target_modules = lora_cfg.get("target_modules")
    _assert_target_modules_exist(model, target_modules)

    peft_cfg = LoraConfig(
        r=lora_cfg["r"],
        lora_alpha=lora_cfg.get("lora_alpha", 2 * lora_cfg["r"]),
        lora_dropout=lora_cfg.get("lora_dropout", 0.05),
        bias=lora_cfg.get("bias", "none"),
        task_type=lora_cfg.get("task_type", "CAUSAL_LM"),
        target_modules=target_modules,
        use_dora=lora_cfg.get("use_dora", False),
        use_rslora=lora_cfg.get("use_rslora", False),
        init_lora_weights=lora_cfg.get("init_lora_weights", True),
    )
    model = get_peft_model(model, peft_cfg)

    n_adapted = sum(1 for name, _ in model.named_modules() if name.endswith(".lora_A.default"))
    logger.info("DoRA/LoRA gắn vào %d module (base: %s).", n_adapted, base_model_id)
    if n_adapted == 0:
        raise ValueError(
            "Không có module nào được gắn adapter — DoRA sẽ không học gì. "
            "Kiểm tra target_modules trong configs/dora.yaml."
        )
    try:
        model.print_trainable_parameters()
    except Exception:
        pass
    return model, processor, peft_cfg


def load_lora_model(checkpoint: str, device: str = "cpu", merge: bool = True):
    """Nạp adapter đã train để inference. Returns (model, processor)."""
    from peft import PeftModel

    processor = load_processor(checkpoint)

    adapter_cfg = os.path.join(checkpoint, "adapter_config.json")
    if os.path.isfile(adapter_cfg):
        with open(adapter_cfg, "r", encoding="utf-8") as f:
            cfg_data = json.load(f)
            base_id = cfg_data.get("base_model_name_or_path")
            target_mods = cfg_data.get("target_modules", [])
        base = load_base_model(base_id, dtype=torch.float32)
        model = PeftModel.from_pretrained(base, checkpoint)
        if merge:
            if "lm_head" in target_mods and getattr(base.config, "tie_word_embeddings", False):
                logger.warning(
                    "CẢNH BÁO AN TOÀN: Adapter chứa 'lm_head' trong khi model có tie_word_embeddings=True. "
                    "Từ chối merge_and_unload() để tránh làm biến dạng ma trận embedding đầu vào (gây lỗi U+FFFD). "
                    "Mô hình sẽ chạy ở chế độ unmerged (merge=False)."
                )
                merge = False
            else:
                model = model.merge_and_unload()  # gộp adapter -> inference nhanh, 0 overhead
    else:
        # checkpoint đã merge sẵn (không có adapter_config)
        model = _florence2_class().from_pretrained(
            checkpoint, attn_implementation="eager", dtype=torch.float32,
        )
    return model.to(device).eval(), processor
