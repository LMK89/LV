"""
collator.py — Collate function cho fine-tune Florence-2 OCR.

Trả về batch dict khớp với NeSyTrainer.compute_loss: `outputs = model(**inputs)` + `inputs["labels"]`.

Mỗi bản ghi dataset là dict {image, prefix, suffix}. Florence-2 nhận ảnh + prompt
(prefix, vd "<OCR>") ở phía encoder và sinh `suffix` ở decoder; loss tính nội bộ
qua `labels` (pad -> -100 để bỏ khỏi loss).
"""
import logging

from PIL import Image

logger = logging.getLogger(__name__)


def make_collate_fn(processor, device=None, max_length: int = 768):
    pad_id = processor.tokenizer.pad_token_id

    def _load_image(path):
        try:
            return Image.open(path).convert("RGB")
        except Exception:
            # Fallback cuối: ảnh đen. train.py đã LỌC ảnh thiếu trước khi
            # train nên fallback này gần như không kích hoạt (tránh crash lẻ).
            logger.warning("Không mở được ảnh %r — dùng ảnh đen thay thế.", path)
            return Image.new("RGB", (768, 768), color=(0, 0, 0))

    def collate(batch):
        images = [_load_image(r.get("image", "")) for r in batch]
        prompts = [r.get("prefix", "<OCR>") for r in batch]
        answers = [r.get("suffix", "") for r in batch]

        inputs = processor(
            text=prompts, images=images, return_tensors="pt", padding=True,
        )
        labels = processor.tokenizer(
            text=answers, return_tensors="pt", padding="longest",
            max_length=max_length, truncation=True, return_token_type_ids=False,
        ).input_ids
        labels[labels == pad_id] = -100  # bỏ pad khỏi cross-entropy

        out = {
            "input_ids": inputs["input_ids"],
            "attention_mask": inputs["attention_mask"],
            "pixel_values": inputs["pixel_values"],
            "labels": labels,
        }
        if device is not None:
            out = {k: v.to(device) for k, v in out.items()}
        return out

    return collate
