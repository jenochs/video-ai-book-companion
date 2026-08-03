"""Summary visualizations accompanying Chapter 1 of "Video Generation with AI".

Builds the chapter's four-panel learning summary: the quality vs speed
tradeoff of inference steps, how subjective each evaluation metric is,
recommended step settings by use case, and the balance between objective
metrics and subjective judgment in evaluating generated video.
"""

import matplotlib.pyplot as plt


def visualize_learning_summary():
    """Create a visual summary of the chapter's evaluation concepts."""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))

    # Panel 1: quality improves with more inference steps, but so does
    # generation time. Around 30 to 50 steps is a practical sweet spot.
    steps = [25, 35, 50, 75]
    quality = [7, 8.5, 9.2, 9.5]
    gen_time = [2, 3, 4.5, 7]

    ax1_twin = ax1.twinx()
    ax1.plot(steps, quality, "b-o", linewidth=2, markersize=8, label="Quality")
    ax1_twin.plot(steps, gen_time, "r-s", linewidth=2, markersize=8, label="Time")
    ax1.set_xlabel("Inference Steps")
    ax1.set_ylabel("Quality Score", color="b")
    ax1_twin.set_ylabel("Generation Time (min)", color="r")
    ax1.set_title("Quality vs Speed Tradeoff")

    # Panel 2: evaluation metrics ranked by subjectivity. SSIM is fairly
    # objective; prompt adherence depends on human interpretation.
    metrics = ["Consistency\n(SSIM)", "Sharpness", "Motion", "Prompt\nAdherence"]
    subjectivity = [0.3, 0.4, 0.5, 0.9]
    ax2.bar(metrics, subjectivity, color=["green", "yellow", "orange", "red"])
    ax2.set_ylabel("Subjectivity Level")
    ax2.set_title("How Subjective Are Video Metrics?")

    # Panel 3: suggested inference step settings by use case. There is no
    # one-size-fits-all number; match the setting to the goal.
    use_cases = ["Social\nMedia", "Professional\nDemo", "Artistic\nProject", "Quick\nPrototype"]
    recommended_steps = [30, 50, 60, 25]
    ax3.bar(use_cases, recommended_steps, color="skyblue", alpha=0.7)
    ax3.set_ylabel("Recommended Inference Steps")
    ax3.set_title("Settings by Use Case")

    # Panel 4: evaluation blends objective metrics with subjective judgment.
    labels = ["Objective\nMetrics", "Subjective\nJudgment"]
    sizes = [40, 60]
    colors = ["lightblue", "lightcoral"]
    ax4.pie(
        sizes,
        labels=labels,
        colors=colors,
        autopct="%1.0f%%",
        startangle=90,
        textprops={"fontsize": 12},
    )
    ax4.set_title("Ideal Evaluation Approach")

    plt.suptitle("Video Generation: Key Concepts Summary", fontsize=16, fontweight="bold")
    plt.tight_layout()
    plt.show()
    return fig


if __name__ == "__main__":
    visualize_learning_summary()
