"""Inference with fine-tuned CogVideoX weights.

Companion module for Chapter 5, "Fine-Tuning for Specific Video Tasks",
from the book "Video Generation with AI". Wraps the chapter's inference
recipe: load the base CogVideoX pipeline in half precision, attach the
trained LoRA adapter weights, generate a clip, and export it to mp4.
Load order matters: base model first, then LoRA weights.
"""

import torch
from diffusers import CogVideoXPipeline
from diffusers.utils import export_to_video


def load_finetuned_pipeline(base_model_id: str = "THUDM/CogVideoX-2B",
                            lora_path: str = "./outputs/disney_lora/checkpoint-1000",
                            device: str = "cuda") -> CogVideoXPipeline:
    """Load the base CogVideoX pipeline and attach LoRA weights.

    Args:
        base_model_id: HuggingFace model id of the base CogVideoX model
        lora_path: Path to the trained LoRA checkpoint directory
        device: Device to place the pipeline on

    Returns:
        CogVideoXPipeline with LoRA weights loaded
    """
    # Load base model
    pipe = CogVideoXPipeline.from_pretrained(
        base_model_id,
        torch_dtype=torch.float16
    ).to(device)

    # Load LoRA weights (base model first, then adapters)
    pipe.load_lora_weights(lora_path)

    return pipe


def generate_video(pipe: CogVideoXPipeline,
                   prompt: str,
                   output_path: str = "output.mp4",
                   num_frames: int = 49,
                   guidance_scale: float = 6.0,
                   fps: int = 8) -> str:
    """Generate a video with a fine-tuned pipeline and save it to disk.

    Note: num_frames must follow the 8N+1 pattern required by CogVideoX
    (9, 17, 25, 33, 41, 49, ...). 49 frames is the production default.

    Args:
        pipe: Pipeline returned by load_finetuned_pipeline
        prompt: Text prompt describing the video
        output_path: Where to write the mp4 file
        num_frames: Number of frames to generate (8N+1)
        guidance_scale: Classifier-free guidance strength
        fps: Frames per second for the exported video

    Returns:
        Path of the saved video file
    """
    if (num_frames - 1) % 8 != 0:
        raise ValueError(
            f"num_frames must be 8N+1 (9, 17, ..., 49), got {num_frames}"
        )

    video_frames = pipe(
        prompt=prompt,
        num_videos_per_prompt=1,
        num_frames=num_frames,
        guidance_scale=guidance_scale,
    ).frames[0]

    export_to_video(video_frames, output_path, fps=fps)
    return output_path


if __name__ == "__main__":
    pipeline = load_finetuned_pipeline()
    generate_video(
        pipeline,
        prompt="Mickey Mouse steering a steamboat, black and white cartoon style",
        output_path="mickey_steamboat.mp4",
    )
