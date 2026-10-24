"""The paper's SigLIP 2 global encoder adapter; no dense-feature probes."""
from __future__ import annotations

class Siglip2AdapterError(RuntimeError):
    pass


def encode_batch_global(*, model, processor, images, device):
    try:
        import torch

        if not isinstance(images, (list, tuple)) or not images:
            raise ValueError
        inputs = processor(images=list(images), return_tensors="pt")
        if not inputs or len(images) <= 0:
            raise ValueError
        cuda_inputs = {}
        for key in sorted(inputs):
            value = inputs[key]
            if not isinstance(value, torch.Tensor):
                raise ValueError
            cuda_inputs[key] = value.to(device)
        with torch.inference_mode():
            feature_output = model.get_image_features(**cuda_inputs)
        raw = getattr(feature_output, "pooler_output", None)
        if (
            not isinstance(raw, torch.Tensor)
            or raw.ndim != 2
            or raw.shape[0] != len(images)
            or raw.shape[1] <= 0
            or not torch.isfinite(raw).all()
        ):
            raise ValueError
        norms = torch.linalg.vector_norm(raw, dim=-1, keepdim=True)
        if not torch.isfinite(norms).all() or not torch.all(norms > 0):
            raise ValueError
        normalized = raw / norms
        if not torch.isfinite(normalized).all() or not torch.allclose(
            torch.linalg.vector_norm(normalized, dim=-1),
            torch.ones((len(images),), dtype=normalized.dtype, device=normalized.device),
            rtol=1e-5,
            atol=1e-6,
        ):
            raise ValueError
        return cuda_inputs, raw, normalized
    except Exception as error:
        raise Siglip2AdapterError() from error


def encode_single_global(*, model, processor, image, device):
    return encode_batch_global(
        model=model, processor=processor, images=(image,), device=device
    )
