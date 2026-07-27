"""Render selected image patches + their explanations as a numbered overlay.

The image (square-resized to match the model's token grid) gets a translucent
box and number badge on each explained patch; a legend beside it maps every
number to its wrapped explanation text.
"""

from __future__ import annotations

import textwrap
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

_PALETTE = [
    (230, 25, 75), (60, 180, 75), (0, 130, 200), (245, 130, 48),
    (145, 30, 180), (70, 240, 240), (240, 50, 230), (170, 110, 40),
    (0, 128, 128), (128, 0, 0), (0, 0, 128), (128, 128, 0),
]
_FONTS = Path(__file__).with_name("fonts")
_FONT_REGULAR = _FONTS / "DejaVuSans.ttf"
_FONT_BOLD = _FONTS / "DejaVuSans-Bold.ttf"


@dataclass(frozen=True)
class PatchExplanation:
    index: int
    row: int
    column: int
    text: str


def render_overlay(image_path: str, explanations: list[PatchExplanation],
                   grid_side: int, out_path: str, image_size: int = 800,
                   legend_width: int = 820) -> None:
    image = _square_image(image_path, image_size)
    _draw_patches(image, explanations, grid_side, image_size)
    legend = _draw_legend(explanations, legend_width, image_size)
    canvas = Image.new("RGB", (image_size + legend.width, max(image_size, legend.height)), "white")
    canvas.paste(image.convert("RGB"), (0, 0))
    canvas.paste(legend, (image_size, 0))
    canvas.save(out_path)


def _draw_patches(image: Image.Image, explanations, grid_side: int, size: int) -> None:
    cell = size / grid_side
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    badge_font = ImageFont.truetype(_FONT_BOLD, int(cell * 0.5))
    for number, item in enumerate(explanations, start=1):
        color = _PALETTE[(number - 1) % len(_PALETTE)]
        x0, y0 = item.column * cell, item.row * cell
        draw.rectangle([x0, y0, x0 + cell, y0 + cell], fill=color + (70,), outline=color + (255,), width=3)
        _badge(draw, x0 + cell / 2, y0 + cell / 2, cell * 0.32, color, str(number), badge_font)
    image.alpha_composite(overlay)


def _draw_legend(explanations, width: int, min_height: int) -> Image.Image:
    text_font = ImageFont.truetype(_FONT_REGULAR, 19)
    badge_font = ImageFont.truetype(_FONT_BOLD, 18)
    line_height, wrap = 26, 62
    wrapped = [textwrap.fill(item.text, wrap) for item in explanations]
    entries = [(w.count("\n") + 1) for w in wrapped]
    height = max(min_height, 30 + sum(lines * line_height + 22 for lines in entries))

    legend = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(legend)
    y = 24
    for number, (item, text, lines) in enumerate(zip(explanations, wrapped, entries), start=1):
        color = _PALETTE[(number - 1) % len(_PALETTE)]
        _badge(draw, 26, y + 11, 13, color, str(number), badge_font)
        draw.text((52, y), f"patch {item.index}  (row {item.row}, col {item.column})",
                  font=badge_font, fill=color)
        draw.multiline_text((52, y + 24), text, font=text_font, fill=(30, 30, 30), spacing=4)
        y += lines * line_height + 22
    return legend


def _badge(draw: ImageDraw.ImageDraw, cx: float, cy: float, radius: float,
           color: tuple, label: str, font: ImageFont.FreeTypeFont) -> None:
    draw.ellipse([cx - radius, cy - radius, cx + radius, cy + radius], fill=color, outline="white", width=2)
    draw.text((cx, cy), label, font=font, fill="white", anchor="mm")


def _square_image(image_path: str, size: int) -> Image.Image:
    return Image.open(image_path).convert("RGBA").resize((size, size))
