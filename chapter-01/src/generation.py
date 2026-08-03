"""Video generation helpers accompanying Chapter 1 of "Video Generation with AI".

Wraps the CogVideoX pipeline setup and text-to-video generation used in the
chapter's hands-on section: environment checks, model loading with memory
optimizations, and generating plus saving a first AI video.
"""

import time

import torch
from diffusers import CogVideoXPipeline
from diffusers.utils import export_to_video

DEFAULT_MODEL_ID = "THUDM/CogVideoX-2b"  # CogVideoX-5b is a larger option
DEFAULT_NEGATIVE_PROMPT = "blurry, low quality, distorted, text"


def check_environment():
    """Print the computational environment and return True if a GPU is available.

    Video generation requires significant resources, so verify the GPU and
    available VRAM before loading a model.
    """
    print("Video Generation AI Environment")
    print(f"PyTorch: {torch.__version__}")
    gpu_available = torch.cuda.is_available()
    print(f"GPU Available: {gpu_available}")
    if gpu_available:
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        vram_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"VRAM: {vram_gb:.1f}GB")
    else:
        print("No GPU detected - video generation will be slower")
    return gpu_available


def load_pipeline(model_id=DEFAULT_MODEL_ID, torch_dtype=torch.float16):
    """Load the CogVideoX pipeline with memory optimizations applied.

    The optimizations matter: CPU offload moves unused components out of VRAM,
    and VAE tiling and slicing keep decoding within the memory budget of
    consumer GPUs (8 GB or more of VRAM for CogVideoX-2B).
    """
    start_time = time.time()
    pipe = CogVideoXPipeline.from_pretrained(model_id, torch_dtype=torch_dtype)

    pipe.enable_model_cpu_offload()  # Move unused components to CPU
    pipe.vae.enable_tiling()         # Process the VAE in tiles
    pipe.vae.enable_slicing()        # Slice VAE batches

    load_time = time.time() - start_time
    print(f"CogVideoX loaded and optimized in {load_time:.1f}s")
    return pipe


def generate_video(
    pipe,
    prompt,
    negative_prompt=DEFAULT_NEGATIVE_PROMPT,
    num_frames=49,
    height=480,
    width=720,
    num_inference_steps=50,
    guidance_scale=6.0,
    seed=42,
):
    """Generate a video from a text prompt and return the list of frames.

    Defaults match the chapter's first generation: 49 frames is about two
    seconds at 24 fps, 720x480 balances quality and memory, 50 inference
    steps trade time for refinement, and a fixed seed keeps runs repeatable.
    """
    start_time = time.time()
    video_frames = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        num_frames=num_frames,
        height=height,
        width=width,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        generator=torch.Generator().manual_seed(seed),
    ).frames[0]
    generation_time = time.time() - start_time

    print("Generation completed successfully")
    print(f"Time: {generation_time:.1f} seconds")
    print(f"Generated: {len(video_frames)} frames ({len(video_frames) / 24:.1f}s of video)")
    return video_frames


def save_video(video_frames, output_filename="my_first_ai_video.mp4", fps=24):
    """Export generated frames to an MP4 file and return the filename."""
    export_to_video(video_frames, output_filename, fps=fps)
    print(f"Video saved as: {output_filename}")
    return output_filename


if __name__ == "__main__":
    check_environment()
    pipeline = load_pipeline()
    frames = generate_video(
        pipeline,
        prompt=(
            "A majestic golden retriever running through a sunlit meadow, "
            "cinematic quality, slow motion"
        ),
    )
    save_video(frames)
