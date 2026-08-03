"""Capability detection and runtime policy.

Accompanies Chapter 8 (From Models to Systems) of "Video Generation with AI"
(O'Reilly) by Joseph Enochs.

Implements Pattern 2, backend as source of truth: capability decisions are
computed once at startup, stored as a read-only fact, and published to every
other layer. Nothing else in the system decides on its own; it asks.

The decision function mirrors backend/runtime_config/runtime_policy.py from
LTX Desktop (pinned commit 81f7f45 in the chapter).
"""

from __future__ import annotations

import platform
from dataclasses import dataclass


def decide_force_api_generations(
    system: str,
    cuda_available: bool,
    vram_gb: int | None,
) -> bool:
    """Decide whether generations must route to the hosted API.

    Returns True when local inference is not viable: macOS, no CUDA,
    unknown VRAM, or less than 15 GB of VRAM. Unknown platforms fail
    closed to API mode.
    """
    if system == "Darwin":  # Always API-only on macOS
        return True
    if system in ("Windows", "Linux"):
        if not cuda_available:
            return True
        if vram_gb is None:
            return True
        return vram_gb < 15  # <15GB VRAM forces API mode
    return True  # Fail closed for unknown platforms


def detect_environment() -> tuple[str, bool, int | None]:
    """Probe the current machine: (system, cuda_available, vram_gb).

    Uses torch when available and degrades gracefully when it is not,
    so the policy can still be computed on any machine.
    """
    system = platform.system()
    cuda_available = False
    vram_gb: int | None = None
    try:
        import torch  # type: ignore

        cuda_available = torch.cuda.is_available()
        if cuda_available:
            props = torch.cuda.get_device_properties(0)
            vram_gb = int(props.total_memory / (1024**3))
    except Exception:
        pass
    return system, cuda_available, vram_gb


@dataclass(frozen=True)
class RuntimePolicy:
    """The published policy: computed once, read by everything.

    Mirrors the payload of GET /api/runtime-policy in LTX Desktop.
    """

    force_api_generations: bool

    def as_payload(self) -> dict[str, bool]:
        """The JSON body the frontend reads once at startup."""
        return {"force_api_generations": self.force_api_generations}


def build_runtime_policy() -> RuntimePolicy:
    """Compute the policy for the current machine, exactly once."""
    system, cuda_available, vram_gb = detect_environment()
    return RuntimePolicy(
        force_api_generations=decide_force_api_generations(
            system, cuda_available, vram_gb
        )
    )
