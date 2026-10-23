"""Genuine pretrained CLIP; loading is explicit and offline by default."""

import os
from pathlib import Path

import numpy as np


class CLIPEncoder:
    def __init__(
        self,
        model_name="openai/clip-vit-base-patch16",
        allow_download=False,
        cache_dir=None,
    ):
        try:
            import torch
            from transformers import CLIPModel, CLIPProcessor
        except ImportError as error:
            raise RuntimeError(
                "Use notebooks/colab_workbench.ipynb for CLIP, or an existing environment with the optional model dependencies."
            ) from error
        self.torch = torch
        cache_dir = str(
            Path(cache_dir)
            if cache_dir
            else Path(
                os.environ.get(
                    "HF_HUB_CACHE",
                    str(
                        Path(
                            os.environ.get(
                                "HF_HOME", str(Path.home() / ".cache" / "huggingface")
                            )
                        )
                        / "hub"
                    ),
                )
            )
        )
        try:
            self.model = CLIPModel.from_pretrained(
                model_name,
                local_files_only=not allow_download,
                cache_dir=cache_dir,
                use_safetensors=False,
            ).eval()
            self.processor = CLIPProcessor.from_pretrained(
                model_name, local_files_only=not allow_download, cache_dir=cache_dir
            )
        except OSError as error:
            if not allow_download:
                raise RuntimeError(
                    "CLIP weights are unavailable locally. Enable model download explicitly, then load again."
                ) from error
            raise RuntimeError(
                "CLIP model download failed. Check network access and model identifier, then retry."
            ) from error

    def _encode(self, items, images):
        if not items:
            raise ValueError("Provide at least one image or description.")
        outputs = []
        with self.torch.inference_mode():
            for start in range(0, len(items), 16):
                batch = items[start : start + 16]
                if images:
                    inputs = self.processor(
                        images=[im.convert("RGB") for im in batch], return_tensors="pt"
                    )
                    features = self.model.get_image_features(**inputs)
                else:
                    if any(not s.strip() for s in batch):
                        raise ValueError("Descriptions cannot be empty.")
                    inputs = self.processor(
                        text=batch, padding=True, truncation=True, return_tensors="pt"
                    )
                    features = self.model.get_text_features(**inputs)
                if not self.torch.is_tensor(features):
                    pooled = getattr(features, "pooler_output", None)
                    if pooled is None:
                        raise RuntimeError(
                            "Unsupported CLIP feature output; use pinned model dependencies."
                        )
                    projection = (
                        self.model.visual_projection
                        if images
                        else self.model.text_projection
                    )
                    features = projection(pooled)
                outputs.append(features.cpu().numpy())
        values = np.concatenate(outputs)
        return values / np.maximum(np.linalg.norm(values, axis=1, keepdims=True), 1e-12)

    def encode_images(self, images):
        return self._encode(images, True)

    def encode_texts(self, texts):
        return self._encode(texts, False)
