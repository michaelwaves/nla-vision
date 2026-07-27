"""Parse an existing NLA checkpoint's nla_meta.yaml sidecar — reuses the
checkpoint's OWN injection scheme (token, neighbors, prompt template, target
scale) instead of adding a fresh special token, so the warm-started model
exercises the exact association it was already RL'd on."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True)
class Sidecar:
    d_model: int
    injection_scale: float
    injection_char: str
    injection_token_id: int
    left_neighbor_id: int
    right_neighbor_id: int
    av_prompt_template: str


def load_sidecar(checkpoint_dir: str | Path, tokenizer) -> Sidecar:
    meta = yaml.safe_load((Path(checkpoint_dir) / "nla_meta.yaml").read_text())
    t = meta["tokens"]
    sidecar = Sidecar(
        d_model=meta["d_model"],
        injection_scale=float(meta["extraction"]["injection_scale"]),
        injection_char=t["injection_char"],
        injection_token_id=t["injection_token_id"],
        left_neighbor_id=t["injection_left_neighbor_id"],
        right_neighbor_id=t["injection_right_neighbor_id"],
        av_prompt_template=meta["prompt_templates"]["av"],
    )
    _assert_matches_live_tokenizer(sidecar, tokenizer)
    return sidecar


def _assert_matches_live_tokenizer(sidecar: Sidecar, tokenizer) -> None:
    """Same two checks nla_inference.py's load_nla_config does — catch
    tokenizer/template drift before training silently learns the wrong
    injection position."""
    live_inj = tokenizer.encode(sidecar.injection_char, add_special_tokens=False)
    assert live_inj == [sidecar.injection_token_id], (
        f"tokenizer drift: {sidecar.injection_char!r} -> {live_inj}, "
        f"sidecar says [{sidecar.injection_token_id}]"
    )
    content = sidecar.av_prompt_template.format(injection_char=sidecar.injection_char)
    ids = tokenizer.apply_chat_template(
        [{"role": "user", "content": content}], tokenize=True, add_generation_prompt=True
    )
    ids = ids.input_ids if hasattr(ids, "input_ids") else ids
    matches = [i for i, tok in enumerate(ids) if tok == sidecar.injection_token_id]
    assert len(matches) == 1, f"injection token appears {len(matches)}x in canonical prompt"
    p = matches[0]
    assert ids[p - 1] == sidecar.left_neighbor_id, f"left neighbor drift: {ids[p-1]}"
    assert ids[p + 1] == sidecar.right_neighbor_id, f"right neighbor drift: {ids[p+1]}"
