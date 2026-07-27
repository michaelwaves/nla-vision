"""Decode one injected vector to its <explanation> text via the frozen actor.

The vector is injected in place of the sidecar's marker token and normalized
to injection_scale by inject_at_token — so it must already live in the actor's
d_model space (a raw Gemma3 residual-stream activation, or a projected one).
"""

from __future__ import annotations

import re

import torch

from nla_vision.injection import inject_at_token
from nla_vision.sidecar import Sidecar

_EXPLANATION_RE = re.compile(r"<explanation>\s*(.*?)\s*</explanation>", re.DOTALL)


@torch.no_grad()
def verbalize(tokenizer, model, sidecar: Sidecar, vector: torch.Tensor,
              max_new_tokens: int = 60) -> str:
    # Derived, not passed: under device_map="auto" the model is sharded and a
    # bare "cuda" would land the injected vector on the wrong shard.
    device = model.get_input_embeddings().weight.device
    marker_count = vector.shape[0] if vector.dim() == 2 else 1
    content = sidecar.av_prompt_template.format(injection_char=sidecar.injection_char * marker_count)
    prompt_ids = tokenizer.apply_chat_template(
        [{"role": "user", "content": content}], tokenize=True, add_generation_prompt=True
    )
    prompt_ids = prompt_ids.input_ids if hasattr(prompt_ids, "input_ids") else prompt_ids
    prompt_ids = torch.tensor(prompt_ids, device=device)
    embeds = model.get_input_embeddings()(prompt_ids)
    embeds = inject_at_token(
        embeds, prompt_ids, sidecar.injection_token_id, vector.to(device), sidecar.injection_scale
    )
    out_ids = model.generate(
        inputs_embeds=embeds.unsqueeze(0),
        attention_mask=torch.ones(1, embeds.shape[0], device=device),
        max_new_tokens=max_new_tokens, do_sample=False,
    )
    text = tokenizer.decode(out_ids[0], skip_special_tokens=True)
    match = _EXPLANATION_RE.search(text)
    return match.group(1).strip() if match else text.strip()
