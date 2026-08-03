"""Structured experiments accompanying Chapter 1 of "Video Generation with AI".

Reproduces the chapter's two guided experiments: how the number of inference
steps affects quality, speed, and motion, and how prompt style (minimal,
detailed, artistic) shapes what the model generates. Both take a loaded
CogVideoX pipeline, for example from generation.load_pipeline().
"""

import time

import torch
from diffusers.utils import export_to_video

from evaluation import evaluate_video_basics


def experiment_inference_steps(pipe):
    """Compare 25 vs 50 inference steps on the same prompt and seed.

    Isolates the effect of refinement steps: more steps usually buy sharper
    detail and more convincing motion at the cost of generation time.
    Returns a list of result dicts including the evaluation metrics.
    """
    test_prompt = "A red rose blooming in timelapse, nature documentary style"
    experiments = [
        {"steps": 25, "name": "Fast (25 steps)", "expectation": "Quick but potentially less refined"},
        {"steps": 50, "name": "Balanced (50 steps)", "expectation": "Good quality/speed tradeoff"},
    ]
    results = []

    for exp in experiments:
        print(f"\nTesting: {exp['name']}")
        print(f"   Expectation: {exp['expectation']}")

        start_time = time.time()
        video = pipe(
            prompt=test_prompt,
            num_frames=25,  # Shorter clip for faster experimentation
            height=480,
            width=720,
            num_inference_steps=exp["steps"],
            guidance_scale=6.0,
            generator=torch.Generator().manual_seed(123),
        ).frames[0]
        gen_time = time.time() - start_time

        filename = f"experiment_steps_{exp['steps']}.mp4"
        export_to_video(video, filename, fps=24)

        print(f"   Generation time: {gen_time:.1f}s")
        print(f"   Saved as: {filename}")

        eval_result = evaluate_video_basics(filename)
        results.append(
            {
                "name": exp["name"],
                "steps": exp["steps"],
                "time": gen_time,
                "evaluation": eval_result,
            }
        )

    print("\nResults Comparison:")
    print("Setting         | Time    | Consistency | Sharpness | Motion")
    print("-" * 60)
    for r in results:
        e = r["evaluation"]
        print(
            f"{r['name']:<15} | {r['time']:>6.1f}s | {e['consistency']:>10.3f} | "
            f"{e['sharpness']:>9.0f} | {e['motion']:>6.3f}"
        )

    return results


def experiment_prompt_styles(pipe):
    """Explore how prompt engineering affects video generation.

    Keeps the subject (a butterfly), settings, and seed fixed while varying
    only the prompt style, from minimal to detailed to artistic. Returns the
    list of generated filenames for side-by-side review.
    """
    print("\nExperiment 2: Prompt Engineering Impact")
    print("=" * 45)
    print("How do different prompt styles affect the generated video?\n")

    prompt_styles = [
        {
            "prompt": "a butterfly",
            "style": "Minimal",
            "expectation": "AI has maximum creative freedom",
        },
        {
            "prompt": "a colorful butterfly landing on a flower, high quality",
            "style": "Detailed",
            "expectation": "More control over the output",
        },
        {
            "prompt": "ethereal glowing butterfly in magical forest, cinematic lighting, fantasy style",
            "style": "Artistic",
            "expectation": "Creative and stylized result",
        },
    ]

    print("Testing different prompt styles with the same subject...\n")
    filenames = []

    for i, style in enumerate(prompt_styles):
        print(f"Style {i + 1}: {style['style']}")
        print(f"   Prompt: \"{style['prompt']}\"")
        print(f"   Expectation: {style['expectation']}")

        video = pipe(
            prompt=style["prompt"],
            num_frames=25,
            height=480,
            width=720,
            num_inference_steps=30,
            guidance_scale=6.0,
            generator=torch.Generator().manual_seed(456),
        ).frames[0]

        filename = f"experiment_prompt_{i + 1}_{style['style'].lower()}.mp4"
        export_to_video(video, filename, fps=24)
        filenames.append(filename)

        print(f"   Generated: {filename}")
        print()

    return filenames


if __name__ == "__main__":
    from generation import load_pipeline

    pipeline = load_pipeline()
    experiment_inference_steps(pipeline)
    experiment_prompt_styles(pipeline)
