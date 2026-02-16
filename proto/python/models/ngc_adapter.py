#!/usr/bin/env python3
"""
Version: 0.9.3
License: MIT
Code generated with support from CODEX and CODEX CLI.
Owner / Idea / Management: Dr. Babak Sorkhpour (https://x.com/Drbabakskr)
Author: Dr. Babak Sorkhpour with support from ChatGPT
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class NGCModelConfig:
    name: str
    artifact_path: Path
    license_note: str


class OptionalNGCAdapter:
    """Optional adapter for locally-provisioned NVIDIA NGC artifacts.

    The repository does not ship NGC binaries. Operators must provide model
    artifacts locally and accept applicable NGC/TAO terms before use.
    """

    def __init__(self, config: NGCModelConfig) -> None:
        self.config = config

    def is_ready(self) -> bool:
        return self.config.artifact_path.exists() and self.config.artifact_path.is_file()

    def load(self) -> str:
        if not self.is_ready():
            raise FileNotFoundError(
                f"Model artifact not found at {self.config.artifact_path}. "
                "Download from NVIDIA NGC locally and keep binaries out of git."
            )
        return f"Loaded optional NGC model: {self.config.name}"
