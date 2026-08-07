from __future__ import annotations

import numpy as np
import open_clip
import torch
from PIL import Image


class ClipEncoder:
    def __init__(
        self,
        model_name: str = "ViT-B-32",
        pretrained: str = "laion2b_s34b_b79k",
        device: str = "cuda",
        fp16: bool = True,
    ):
        self._device = device
        self._fp16 = fp16
        self._model, _, self._preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained, device=device,
        )
        if fp16:
            self._model = self._model.half()
        self._model.eval()
        self._tokenizer = open_clip.get_tokenizer(model_name)

    @property
    def embed_dim(self) -> int:
        return self._model.visual.output_dim

    def encode_image(self, crops: list[np.ndarray]) -> np.ndarray:
        """Encode a list of RGB crops to L2-normalized embeddings [N, D]."""
        if not crops:
            return np.empty((0, self.embed_dim), dtype=np.float32)

        images = []
        for crop in crops:
            pil_img = Image.fromarray(crop)
            images.append(self._preprocess(pil_img))

        batch = torch.stack(images).to(self._device)
        if self._fp16:
            batch = batch.half()

        with torch.no_grad():
            embeds = self._model.encode_image(batch)

        embeds = embeds.float()
        embeds = embeds / embeds.norm(dim=-1, keepdim=True)
        return embeds.cpu().numpy()

    def encode_text(self, prompts: list[str]) -> np.ndarray:
        """Encode a list of text prompts to L2-normalized embeddings [N, D]."""
        if not prompts:
            return np.empty((0, self.embed_dim), dtype=np.float32)

        tokens = self._tokenizer(prompts).to(self._device)
        with torch.no_grad():
            embeds = self._model.encode_text(tokens)

        embeds = embeds.float()
        embeds = embeds / embeds.norm(dim=-1, keepdim=True)
        return embeds.cpu().numpy()
