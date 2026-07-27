"""Filesystem layout of one run's outputs, rooted at a single directory."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class RunPaths:
    directory: Path

    def __post_init__(self) -> None:
        self.directory.mkdir(parents=True, exist_ok=True)

    @property
    def activations(self) -> Path:
        return self.directory / "acts.pt"

    @property
    def overlay(self) -> Path:
        return self.directory / "overlay.png"

    @property
    def viewer(self) -> Path:
        return self.directory / "viewer.html"
