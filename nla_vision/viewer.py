"""Build a self-contained interactive HTML viewer: the image, a per-token norm
heatmap you can toggle, and a hover tooltip showing each patch's explanation.
"""

from __future__ import annotations

import base64
import json
import mimetypes
from pathlib import Path

from nla_vision.overlay import PatchExplanation

_TEMPLATE = Path(__file__).with_name("viewer_template.html")
# blue sequential ramp (dataviz reference palette), light -> dark = low -> high norm
_RAMP = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]


def render_html(image_path: str, grid_side: int, norms: list[float],
                explanations: list[PatchExplanation], out_path: str) -> None:
    payload = _build_payload(grid_side, norms, explanations)
    html = (_TEMPLATE.read_text()
            .replace("__PAYLOAD__", json.dumps(payload))
            .replace("__IMAGE_SRC__", _data_uri(image_path)))
    Path(out_path).write_text(html)


def _build_payload(grid_side: int, norms: list[float],
                   explanations: list[PatchExplanation]) -> dict:
    text_by_index = {item.index: item.text for item in explanations}
    low, high = _robust_range(norms)
    span = high - low or 1.0
    tokens = [{
        "index": index,
        "row": index // grid_side,
        "col": index % grid_side,
        "norm": round(float(norm)),
        "t": round(min(1.0, max(0.0, (float(norm) - low) / span)), 3),
        "text": text_by_index.get(index),
    } for index, norm in enumerate(norms)]
    return {"grid_side": grid_side, "ramp": _RAMP, "tokens": tokens}


def _robust_range(norms: list[float]) -> tuple[float, float]:
    """5th/95th percentile — one sink-token outlier must not flatten the ramp."""
    values = sorted(float(n) for n in norms)
    return values[int(0.05 * (len(values) - 1))], values[int(0.95 * (len(values) - 1))]


def _data_uri(image_path: str) -> str:
    mime = mimetypes.guess_type(image_path)[0] or "image/jpeg"
    encoded = base64.b64encode(Path(image_path).read_bytes()).decode()
    return f"data:{mime};base64,{encoded}"
