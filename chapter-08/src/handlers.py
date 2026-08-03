"""Lock-aware handlers, the TaskRunner, and the capstone audio-to-video handler.

Accompanies Chapter 8 (From Models to Systems) of "Video Generation with AI"
(O'Reilly) by Joseph Enochs.

Implements Pattern 8, lock-aware handlers: state mutation happens under an
explicit RLock via the @with_state_lock decorator, and heavy work is
submitted to a TaskRunner thread that never holds the lock. The
AudioToVideoHandler at the bottom is the chapter's capstone, the LTX Desktop
side of exposing the audio-to-video pipeline, wired against the service
Protocols so it runs against fakes on any machine.

Mirrors backend/handlers/base.py and the capstone handler
backend/handlers/audio_to_video_handler.py (companion @ ch8-capstone).
"""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime
from functools import wraps
from pathlib import Path
from uuid import UUID, uuid4

from services import ServiceBundle
from state_machine import (
    GenerationComplete,
    GenerationError,
    GenerationPending,
    GenerationRunning,
    GenerationState,
)


class GenerationAlreadyRunningError(RuntimeError):
    """Raised when a generation is requested while one is in flight."""


@dataclass
class StateProvider:
    """Minimal shared state container; LTX Desktop's is richer."""

    current_generation: GenerationState | None = None


class TaskRunner:
    """Thin wrapper around a thread pool: lifecycle, exception capture, IDs.

    Handlers always submit long work here; the runner never holds the
    handler's state lock.
    """

    def __init__(self, max_workers: int = 2):
        self._pool = ThreadPoolExecutor(max_workers=max_workers)
        self._futures: dict[UUID, object] = {}

    def submit(self, fn, *args, **kwargs) -> UUID:
        task_id = uuid4()
        self._futures[task_id] = self._pool.submit(fn, *args, **kwargs)
        return task_id

    def wait_all(self, timeout: float | None = None) -> None:
        for future in list(self._futures.values()):
            future.result(timeout=timeout)  # re-raises captured exceptions

    def shutdown(self) -> None:
        self._pool.shutdown(wait=True)


class HandlerBase:
    """Base class: every handler gets a state provider and a reentrant lock."""

    def __init__(self, state_provider: StateProvider):
        self._state = state_provider
        self._lock = threading.RLock()


def with_state_lock(method):
    """Acquire the handler's RLock for the duration of the method."""

    @wraps(method)
    def wrapper(self, *args, **kwargs):
        with self._lock:
            return method(self, *args, **kwargs)

    return wrapper


# ---------------------------------------------------------------------------
# Capstone: the audio-to-video handler (LTX Desktop implementation)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class A2VidRequest:
    """Typed request body for POST /api/audio-to-video."""

    audio_path: str
    prompt: str = ""
    duration_s: float = 5.0
    size: tuple[int, int] = (768, 512)


class AudioToVideoHandler(HandlerBase):
    """Expose the audio-to-video pipeline behind the lock-aware discipline.

    start() holds the lock briefly for state mutation only, submits the
    heavy work to the TaskRunner, and returns a job ID immediately. The
    background _run() acquires the lock only for the short state writes.
    """

    def __init__(
        self,
        services: ServiceBundle,
        state_provider: StateProvider,
        task_runner: TaskRunner,
    ):
        super().__init__(state_provider)
        self._services = services
        self._task_runner = task_runner

    @with_state_lock
    def start(self, request: A2VidRequest) -> UUID:
        # Holds the lock briefly for state mutation only.
        if self._state.current_generation is not None and isinstance(
            self._state.current_generation, (GenerationPending, GenerationRunning)
        ):
            raise GenerationAlreadyRunningError()
        job_id = uuid4()
        self._state.current_generation = GenerationPending(
            job_id=job_id,
            submitted_at=datetime.utcnow(),
        )
        # Lock will release when this method returns.
        self._task_runner.submit(self._run, job_id, request)
        return job_id

    @with_state_lock
    def snapshot(self) -> GenerationState | None:
        """What the renderer polls: the current state, read under the lock."""
        return self._state.current_generation

    def _run(self, job_id: UUID, request: A2VidRequest) -> None:
        # Runs in a background thread. Does NOT hold the lock.
        started = datetime.utcnow()

        def on_progress(progress: float, step: int) -> None:
            with self._lock:
                self._state.current_generation = GenerationRunning(
                    job_id=job_id,
                    started_at=started,
                    progress=progress,
                    current_step=step,
                    estimated_completion=None,
                )

        try:
            output_path: Path = self._services.fast_video_pipeline(
                prompt=request.prompt,
                image=None,
                size=request.size,
                duration_seconds=request.duration_s,
                on_progress=on_progress,
            )
            with self._lock:
                self._state.current_generation = GenerationComplete(
                    job_id=job_id,
                    completed_at=datetime.utcnow(),
                    output_path=str(output_path),
                    duration_seconds=request.duration_s,
                )
        except Exception as e:
            with self._lock:
                self._state.current_generation = GenerationError(
                    job_id=job_id,
                    failed_at=datetime.utcnow(),
                    error_message=str(e),
                    error_kind=type(e).__name__,
                )
