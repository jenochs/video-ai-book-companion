"""State as discriminated unions.

Accompanies Chapter 8 (From Models to Systems) of "Video Generation with AI"
(O'Reilly) by Joseph Enochs.

Implements Patterns 5 and 6: the DownloadingSession shape used for per-file
download progress, and the five-variant GenerationState union that makes
illegal states unrepresentable. Each variant carries only the fields that are
meaningful to its phase, and handlers consume the union with exhaustive
structural pattern matching (Python 3.10+).

Mirrors backend/state/app_state_types.py from LTX Desktop (pinned commit
81f7f45 in the chapter).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Union
from uuid import UUID

# ---------------------------------------------------------------------------
# Pattern 5: per-file download progress
# ---------------------------------------------------------------------------

# In LTX Desktop this is an enum of model file types (checkpoint, upsampler,
# text encoder, ...). A plain string keeps the companion dependency-free.
ModelFileType = str


@dataclass(frozen=True)
class DownloadingSession:
    """Per-file download state: every file is pending, in progress, or done."""

    session_id: UUID
    requested_files: tuple[ModelFileType, ...]
    completed_files: frozenset[ModelFileType]
    current_file: ModelFileType | None
    bytes_downloaded: int
    bytes_total: int
    download_speed_bps: float
    started_at: datetime

    def file_status(self, file_type: ModelFileType) -> str:
        """Classify one requested file. Every file is in exactly one bucket."""
        if file_type in self.completed_files:
            return "done"
        if file_type == self.current_file:
            return "in progress"
        return "pending"


# ---------------------------------------------------------------------------
# Pattern 6: GenerationState as a discriminated union
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GenerationPending:
    job_id: UUID
    submitted_at: datetime


@dataclass(frozen=True)
class GenerationRunning:
    job_id: UUID
    started_at: datetime
    progress: float  # always present, 0.0 - 1.0
    current_step: int  # always present
    estimated_completion: datetime | None


@dataclass(frozen=True)
class GenerationComplete:
    job_id: UUID
    completed_at: datetime
    output_path: str
    duration_seconds: float


@dataclass(frozen=True)
class GenerationError:
    job_id: UUID
    failed_at: datetime
    error_message: str
    error_kind: str


@dataclass(frozen=True)
class GenerationCancelled:
    job_id: UUID
    cancelled_at: datetime
    cancelled_at_step: int


GenerationState = Union[
    GenerationPending,
    GenerationRunning,
    GenerationComplete,
    GenerationError,
    GenerationCancelled,
]


def describe_generation_state(state: GenerationState) -> dict:
    """Handler-style exhaustive match over the union.

    Add a sixth variant and a strict type checker (pyright, mypy) flags
    every match like this one as incomplete.
    """
    match state:
        case GenerationPending(job_id=jid):
            return {"status": "queued", "job": str(jid)}
        case GenerationRunning(progress=p, current_step=step):
            return {"status": "running", "progress": p, "step": step}
        case GenerationComplete(output_path=path, duration_seconds=dur):
            return {"status": "done", "output": path, "elapsed_s": dur}
        case GenerationError(error_message=msg, error_kind=kind):
            return {"status": "error", "kind": kind, "message": msg}
        case GenerationCancelled(cancelled_at_step=step):
            return {"status": "cancelled", "at_step": step}
    raise TypeError(f"Unhandled state: {state!r}")
