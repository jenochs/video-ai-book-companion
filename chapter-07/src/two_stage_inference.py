"""Two-stage flow matching inference utilities for Chapter 7 of Video Generation with AI.

This module accompanies Chapter 7, "Audio-Visual Synchronization with
Cross-Modal Fusion". It implements the inference-side machinery of the
LTX-2 two-stage pipeline in a compact form: classifier-free guidance,
the Euler diffusion step, the joint audio-video denoising loop, sigma
schedules for both stages, and the frames % 8 == 1 constraint check.

These pieces follow ltx_core/components/guiders.py and
ltx_pipelines/utils/helpers.py. They are model-agnostic: any callable
that maps (video_state, audio_state, sigmas, step_index) to denoised
predictions can drive the loop, which is how the notebook demos joint
denoising with a toy model on CPU.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Callable, Optional, Tuple

import torch
from torch import Tensor

# Stage 2 uses only 4 distilled sigma values instead of 40 steps.
STAGE_2_DISTILLED_SIGMA_VALUES = [0.909375, 0.725, 0.421875, 0.0]

# Valid LTX-2 frame counts satisfy frames % 8 == 1.
FRAME_MODULUS = 8


def validate_num_frames(num_frames: int) -> int:
    """Check the LTX-2 frame-count constraint: frames % 8 == 1.

    Valid values include 1, 9, 17, 25, ..., 121. Raises ValueError with
    the nearest valid choices if the constraint is violated.
    """
    if num_frames < 1 or num_frames % FRAME_MODULUS != 1:
        lower = max(1, ((num_frames - 1) // FRAME_MODULUS) * FRAME_MODULUS + 1)
        upper = lower + FRAME_MODULUS
        raise ValueError(
            f"num_frames must satisfy frames % 8 == 1; got {num_frames}. "
            f"Nearest valid values: {lower} or {upper}."
        )
    return num_frames


@dataclass(frozen=True)
class LatentState:
    """Latent tensor plus optional conditioning info for one modality."""

    latent: Tensor
    denoise_mask: Optional[Tensor] = None  # 1.0 where denoising applies
    clean_latent: Optional[Tensor] = None  # Known-clean content to keep


@dataclass(frozen=True)
class CFGGuider:
    """Classifier-free guidance guider.

    Amplifies the difference between the conditioned and unconditioned
    predictions: output = cond + (scale - 1) * (cond - uncond). LTX-2
    Stage 1 defaults to scale=4.0; Stage 2 disables guidance entirely.
    """

    scale: float = 4.0

    def delta(self, cond: Tensor, uncond: Tensor) -> Tensor:
        """The guidance correction added to the conditioned prediction."""
        return (self.scale - 1) * (cond - uncond)

    def enabled(self) -> bool:
        return self.scale != 1.0

    def apply(self, cond: Tensor, uncond: Tensor) -> Tensor:
        """Full CFG combination: cond + delta(cond, uncond)."""
        if not self.enabled():
            return cond
        return cond + self.delta(cond, uncond)


class EulerDiffusionStep:
    """One Euler step of the flow matching ODE.

    With an x0-parameterized model, the velocity toward the clean sample
    is (x_t - x0_pred) / sigma_t, and the update moves along it:
    x_{t+1} = x_t + (sigma_{t+1} - sigma_t) * velocity.
    """

    def step(
        self, latent: Tensor, denoised: Tensor, sigmas: Tensor, step_index: int
    ) -> Tensor:
        sigma_now = sigmas[step_index]
        sigma_next = sigmas[step_index + 1]
        if float(sigma_now) == 0.0:
            return denoised
        velocity = (latent - denoised) / sigma_now
        return latent + (sigma_next - sigma_now) * velocity


def make_sigma_schedule(steps: int = 40) -> Tensor:
    """Stage 1 style sigma schedule: steps+1 values from 1.0 down to 0.0.

    The production LTX2Scheduler shapes this curve more carefully, but a
    linear ramp reproduces the structure: start at pure noise
    (sigma = 1.0) and end fully denoised (sigma = 0.0).
    """
    return torch.linspace(1.0, 0.0, steps + 1)


def post_process_latent(
    denoised: Tensor,
    denoise_mask: Optional[Tensor],
    clean_latent: Optional[Tensor],
) -> Tensor:
    """Blend the model prediction with known-clean content.

    Where the mask is 0 (conditioning frames, keyframes), the clean
    latent is kept; where it is 1, the denoised prediction is used.
    """
    if denoise_mask is None or clean_latent is None:
        return denoised
    return denoise_mask * denoised + (1 - denoise_mask) * clean_latent


DenoisingFunc = Callable[
    [LatentState, LatentState, Tensor, int], Tuple[Tensor, Tensor]
]


def euler_denoising_loop(
    sigmas: Tensor,
    video_state: LatentState,
    audio_state: LatentState,
    stepper: EulerDiffusionStep,
    denoise_fn: DenoisingFunc,
    verbose: bool = False,
) -> Tuple[LatentState, LatentState]:
    """Joint audio-video denoising over a diffusion schedule.

    Both modalities are updated together at every step, which is what
    preserves audio-video alignment as noise is removed. The denoise_fn
    receives both states and returns denoised predictions for both.
    """
    for step_idx in range(len(sigmas) - 1):
        denoised_video, denoised_audio = denoise_fn(
            video_state, audio_state, sigmas, step_idx
        )

        denoised_video = post_process_latent(
            denoised_video, video_state.denoise_mask, video_state.clean_latent
        )
        denoised_audio = post_process_latent(
            denoised_audio, audio_state.denoise_mask, audio_state.clean_latent
        )

        video_state = replace(
            video_state,
            latent=stepper.step(video_state.latent, denoised_video, sigmas, step_idx),
        )
        audio_state = replace(
            audio_state,
            latent=stepper.step(audio_state.latent, denoised_audio, sigmas, step_idx),
        )

        if verbose:
            print(
                f"step {step_idx + 1:>3}/{len(sigmas) - 1}  "
                f"sigma {float(sigmas[step_idx]):.4f} -> {float(sigmas[step_idx + 1]):.4f}"
            )

    return video_state, audio_state


def cfg_denoising_func(
    model_fn: Callable[[Tensor, Tensor, Tensor, str], Tuple[Tensor, Tensor]],
    guider: CFGGuider,
) -> DenoisingFunc:
    """Wrap a toy model into a CFG-guided denoising function.

    model_fn(video_latent, audio_latent, sigma, prompt_kind) must return
    (video_pred, audio_pred), where prompt_kind is "positive" or
    "negative". This mirrors LTX-2's guider_denoising_func: a forward
    pass per prompt, combined through the guider.
    """

    def denoise(
        video_state: LatentState,
        audio_state: LatentState,
        sigmas: Tensor,
        step_index: int,
    ) -> Tuple[Tensor, Tensor]:
        sigma = sigmas[step_index]
        pos_video, pos_audio = model_fn(
            video_state.latent, audio_state.latent, sigma, "positive"
        )
        if not guider.enabled():
            return pos_video, pos_audio

        neg_video, neg_audio = model_fn(
            video_state.latent, audio_state.latent, sigma, "negative"
        )
        return guider.apply(pos_video, neg_video), guider.apply(pos_audio, neg_audio)

    return denoise
