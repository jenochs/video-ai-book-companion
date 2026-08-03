"""Service Protocols, fakes, and the ServiceBundle.

Accompanies Chapter 8 (From Models to Systems) of "Video Generation with AI"
(O'Reilly) by Joseph Enochs.

Implements Pattern 7, service Protocols and fakes (no mocks): every side
effect hides behind a typing.Protocol, production wires in real
implementations, tests and notebook demos wire in hand-written fakes that
satisfy the same contract.

Mirrors backend/services/interfaces.py, backend/tests/fakes.py, and
backend/services/bundle.py from LTX Desktop (pinned commit 81f7f45 in the
chapter). The real LTXFastVideoPipeline needs the LTX-2 checkpoint and a
large GPU, so the fakes here also let the whole notebook run on CPU.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Protocol

# ---------------------------------------------------------------------------
# The Protocols: what the rest of the code can ask for
# ---------------------------------------------------------------------------


class FastVideoPipeline(Protocol):
    """Contract for a text or audio conditioned video generation pipeline."""

    def __call__(
        self,
        prompt: str,
        image: object | None,
        size: tuple[int, int],
        duration_seconds: float,
        on_progress: Callable[[float, int], None],
    ) -> Path: ...


class ModelDownloader(Protocol):
    """Contract for fetching one model file with progress reporting."""

    def download(
        self,
        file_type: str,
        target_path: Path,
        on_progress: Callable[[int, int, float], None],
    ) -> None: ...


class GpuInfo(Protocol):
    """Contract for GPU capability probing."""

    @property
    def cuda_available(self) -> bool: ...

    @property
    def vram_gb(self) -> int | None: ...


# ---------------------------------------------------------------------------
# Hand-written fakes: type-checked against the same Protocols
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FakeCall:
    """One recorded pipeline invocation, for later assertions."""

    prompt: str
    size: tuple[int, int]
    duration: float


class FakeFastVideoPipeline:
    """Test double: returns a tiny canned video file after a short delay.

    Records every call, emits realistic progress callbacks, and returns a
    real file path so downstream code can run end to end.
    """

    def __init__(self, fixture_video_path: Path | None = None):
        if fixture_video_path is None:
            fixture_video_path = Path("fixture_tiny.mp4")
            if not fixture_video_path.exists():
                # Minimal placeholder bytes standing in for a tiny MP4 fixture.
                fixture_video_path.write_bytes(b"\x00\x00\x00\x18ftypmp42")
        self.fixture = fixture_video_path
        self.calls: list[FakeCall] = []  # tests can assert on this

    def __call__(self, prompt, image, size, duration_seconds, on_progress) -> Path:
        self.calls.append(
            FakeCall(prompt=prompt, size=size, duration=duration_seconds)
        )
        for step in range(0, 100, 10):
            on_progress(step / 100.0, step)
            time.sleep(0.01)
        on_progress(1.0, 100)
        return self.fixture


class FakeModelDownloader:
    """Test double: simulates a chunked download without touching the network."""

    def __init__(self, total_bytes: int = 1_000_000, chunks: int = 10):
        self.total_bytes = total_bytes
        self.chunks = chunks
        self.downloads: list[str] = []

    def download(self, file_type, target_path, on_progress) -> None:
        self.downloads.append(file_type)
        chunk = self.total_bytes // self.chunks
        for i in range(1, self.chunks + 1):
            on_progress(i * chunk, self.total_bytes, 50e6)
            time.sleep(0.005)
        Path(target_path).write_bytes(b"\x00" * 16)  # tiny stand-in file


@dataclass(frozen=True)
class FakeGpuInfo:
    """Test double: a GPU report you choose, not the machine you happen to have."""

    cuda_available: bool = True
    vram_gb: int | None = 24


# ---------------------------------------------------------------------------
# The ServiceBundle: composition seam for production and tests
# ---------------------------------------------------------------------------


@dataclass
class ServiceBundle:
    """Holds every side-effect Protocol; handlers receive it by injection.

    Production startup builds it with real implementations; tests and
    notebook demos build it with fakes. Handlers cannot tell the difference,
    which is the point. LTX Desktop's bundle holds about 12 services; this
    companion version keeps the three the capstone needs.
    """

    fast_video_pipeline: FastVideoPipeline
    model_downloader: ModelDownloader
    gpu_info: GpuInfo


def build_fake_bundle(**overrides) -> ServiceBundle:
    """Convenience constructor used by the notebook and tests."""
    defaults: dict = dict(
        fast_video_pipeline=FakeFastVideoPipeline(),
        model_downloader=FakeModelDownloader(),
        gpu_info=FakeGpuInfo(),
    )
    defaults.update(overrides)
    return ServiceBundle(**defaults)
