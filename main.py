"""Demo: verbalize Gemma3's own per-image-token activations, no connector.

Gemma3 is natively multimodal. Run an image through it, grab the L32
residual-stream activation at each of the 256 image tokens, then verbalize each
with the frozen Gemma3 NLA actor. Those activations are in-distribution for the
actor (trained on Gemma3 L32 vectors), so no trained projector is needed.

Base model and actor load sequentially — ~24GB each at 12B bfloat16 quantization (one GPU)

Each invocation writes acts.pt, overlay.png and viewer.html to runs/<timestamp>/.

    python main.py IMAGE.jpg --top-k 12
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import click
import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoModelForImageTextToText, AutoProcessor, AutoTokenizer

from nla_vision.overlay import PatchExplanation, render_overlay
from nla_vision.run_paths import RunPaths
from nla_vision.sidecar import load_sidecar
from nla_vision.verbalize import verbalize
from nla_vision.viewer import render_html

_PROJECT_DIR = Path(__file__).parent
_ACTOR = str(_PROJECT_DIR / "checkpoints" / "gemma3_12b_L32_av")
_RUNS_DIR = _PROJECT_DIR / "runs"
_IMAGE_TOKEN_ID = 262144


@dataclass(frozen=True)
class DemoConfig:
    image_path: str
    run_dir: str
    base_model: str = "google/gemma-3-12b-it"
    actor_dir: str = _ACTOR
    layer_index: int = 32
    top_k: int = 12
    drop_massive: int = 0
    max_new_tokens: int = 60
    acts_path: str | None = None
    extract_only: bool = False


@click.command()
@click.argument("image_path")
@click.option("--run-dir", default=None, help="Output directory (default: runs/<timestamp>)")
@click.option("--base-model", default=DemoConfig.base_model, help="Multimodal Gemma3 the actor was trained on")
@click.option("--actor-dir", default=DemoConfig.actor_dir)
@click.option("--layer-index", default=DemoConfig.layer_index, type=int)
@click.option("--top-k", default=DemoConfig.top_k, type=int, help="Salient image tokens to decode; 0 = all 256")
@click.option("--drop-massive", default=DemoConfig.drop_massive, type=int,
              help="Zero the N largest massive-activation dims before selecting/injecting")
@click.option("--acts-path", default=None, help="Reuse an earlier run's acts.pt instead of extracting")
@click.option("--extract-only", is_flag=True, help="Cache activations and exit, freeing VRAM")
def main(image_path, run_dir, base_model, actor_dir, layer_index, top_k, drop_massive,
         acts_path, extract_only):
    run_dir = run_dir or str(
        _RUNS_DIR / datetime.now().strftime("%Y%m%d-%H%M%S"))
    run(DemoConfig(
        image_path=image_path, run_dir=run_dir, base_model=base_model,
        actor_dir=actor_dir, layer_index=layer_index, top_k=top_k, drop_massive=drop_massive,
        acts_path=acts_path, extract_only=extract_only,
    ))


def run(cfg: DemoConfig) -> None:
    paths = RunPaths(Path(cfg.run_dir))
    acts_path = Path(cfg.acts_path) if cfg.acts_path else paths.activations
    if cfg.extract_only or not acts_path.exists():
        _extract_and_cache(cfg, acts_path)
    if cfg.extract_only:
        return
    _verbalize_and_render(cfg, acts_path, paths)


def _verbalize_and_render(cfg: DemoConfig, acts_path: Path, paths: RunPaths) -> None:
    cached = torch.load(acts_path, weights_only=False)
    activations, side, norms = cached["activations"], cached["side"], cached["norms"]
    indices = _select_indices(activations, cfg)

    tokenizer, model, sidecar = _load_actor(cfg)
    explanations = []
    for index in indices.tolist():
        text = verbalize(tokenizer, model, sidecar,
                         activations[index], cfg.max_new_tokens)
        row, column = index // side, index % side
        print(
            f"token {index:3d} (row {row:2d}, col {column:2d}): {text}", flush=True)
        explanations.append(PatchExplanation(
            index=index, row=row, column=column, text=text))

    render_overlay(cfg.image_path, explanations, side, str(paths.overlay))
    render_html(cfg.image_path, side, norms.tolist(),
                explanations, str(paths.viewer))
    print(f"saved run to {paths.directory}")


def _select_indices(activations: torch.Tensor, cfg: DemoConfig) -> torch.Tensor:
    """Top-K image tokens by L2 norm, in spatial order; top_k=0 means all."""
    k = cfg.top_k if cfg.top_k > 0 else activations.shape[0]
    return activations.norm(dim=-1).topk(min(k, activations.shape[0])).indices.sort().values


@torch.no_grad()
def _extract_and_cache(cfg: DemoConfig, acts_path: Path) -> None:
    """Forward the image through multimodal Gemma3, cache every image-token
    activation at layer_index. Selection happens later, off the cache."""
    processor = AutoProcessor.from_pretrained(cfg.base_model)
    model = AutoModelForImageTextToText.from_pretrained(
        cfg.base_model, dtype=torch.bfloat16, device_map="auto",
    ).eval()
    messages = [{"role": "user", "content": [
        {"type": "image", "image": Image.open(cfg.image_path).convert("RGB")},
        {"type": "text", "text": "Describe the image."},
    ]}]
    inputs = processor.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=True,
        return_dict=True, return_tensors="pt",
    ).to(model.device)
    out = model(**inputs, output_hidden_states=True, use_cache=False)
    # output of decoder block layer_index
    hidden = out.hidden_states[cfg.layer_index + 1][0]
    image_mask = inputs["input_ids"][0] == _IMAGE_TOKEN_ID
    image_activations = hidden[image_mask].float().cpu()  # [256, d_model]
    # heatmap shows true norms (sinks included)
    raw_norms = image_activations.norm(dim=-1)

    if cfg.drop_massive:  # zeroing the massive dims only affects selection + injection
        massive_dims = image_activations.abs().mean(0).topk(cfg.drop_massive).indices
        image_activations[:, massive_dims] = 0.0

    side = int(image_activations.shape[0] ** 0.5)
    torch.save({"activations": image_activations,
               "side": side, "norms": raw_norms}, acts_path)
    print(f"saved {image_activations.shape} activations to {acts_path}")

    # Dropping the reference alone leaves VRAM reserved by the caching allocator,
    # and the next device_map="auto" would then offload to CPU.
    del model, out, hidden
    torch.cuda.empty_cache()


def _load_actor(cfg: DemoConfig):
    tokenizer = AutoTokenizer.from_pretrained(cfg.actor_dir)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        cfg.actor_dir, dtype=torch.bfloat16, device_map="auto",
    ).eval()
    sidecar = load_sidecar(cfg.actor_dir, tokenizer)
    return tokenizer, model, sidecar


if __name__ == "__main__":
    main()
