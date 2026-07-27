"""Embedding-injection mechanic: overwrite one token's embedding with an
external vector, mirroring nla_inference.py's approach but standalone (no
sidecar, no fixed architecture-specific scale table — this is a new model).
"""

from __future__ import annotations

import torch


def add_injection_token(tokenizer, model, injection_token: str) -> int:
    """Register injection_token as a special token, resize model embeddings.

    The new row's initial weight is irrelevant — every forward pass
    overwrites it with an external vector before the row is ever read by a
    real attention computation, so it never accumulates a meaningful
    gradient of its own.
    """
    num_added = tokenizer.add_special_tokens({"additional_special_tokens": [injection_token]})
    assert num_added == 1, f"expected to add 1 token, added {num_added}"
    model.resize_token_embeddings(len(tokenizer))
    token_id = tokenizer.convert_tokens_to_ids(injection_token)
    assert token_id is not None and token_id != tokenizer.unk_token_id
    return token_id


def mean_embedding_norm(model) -> float:
    """Average L2 norm of the model's real token embeddings — used as the
    target scale for injected vectors so they're roughly in-distribution."""
    weight = model.get_input_embeddings().weight.detach()
    return weight.float().norm(dim=-1).mean().item()


def build_user_prompt_ids(tokenizer, injection_token: str) -> list[int]:
    """Tokenize the fixed instruction template, generation prompt included."""
    content = f"Describe what this vector represents: {injection_token}"
    messages = [{"role": "user", "content": content}]
    out = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
    # Newer transformers (5.x) returns a BatchEncoding here instead of a plain
    # list of ints; older versions return the list directly.
    return out.input_ids if hasattr(out, "input_ids") else out


def inject_at_token(
    embeds: torch.Tensor,  # [T, d]
    input_ids: torch.Tensor,  # [T]
    injection_token_id: int,
    vector: torch.Tensor,  # [d] whole-image, or [K, d] per-patch; raw scale
    target_scale: float,
) -> torch.Tensor:
    """Rescale each vector to target_scale L2 norm, overwrite embeds at the
    marker positions. Markers and vectors match 1:1 in order — one marker for
    a whole-image vector, K markers for K patch vectors."""
    positions = (input_ids == injection_token_id).nonzero(as_tuple=True)[0]
    vectors = vector if vector.dim() == 2 else vector.unsqueeze(0)
    assert len(positions) == vectors.shape[0], (
        f"expected {vectors.shape[0]} injection positions, found {len(positions)}"
    )
    norms = vectors.float().norm(dim=-1, keepdim=True).clamp_min(1e-12)
    scaled = (vectors.float() / norms * target_scale).to(embeds.dtype)
    out = embeds.clone()
    out[positions] = scaled
    return out
