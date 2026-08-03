"""LoRA fine-tuning and evaluation helpers for Chapter 6 of Video Generation with AI (O'Reilly).

This module accompanies Chapter 6, "Vision Language Models for Video
Understanding". It applies the LoRA techniques from Chapter 5 to Qwen3-VL for
domain adaptation, and includes a simple evaluation framework for video
question answering.
"""

from dataclasses import dataclass
from typing import Dict, List


def build_lora_config(r: int = 64, lora_alpha: int = 128, lora_dropout: float = 0.05):
    """Build a LoRA config targeting the language model's attention layers.

    Trains roughly 0.5% of the model's parameters, the same approach used for
    video generation fine-tuning in Chapter 5.
    """
    from peft import LoraConfig, TaskType

    return LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        bias="none"
    )


def apply_lora(model, lora_config=None):
    """Wrap a base VLM with LoRA adapters and report trainable parameters."""
    from peft import get_peft_model

    if lora_config is None:
        lora_config = build_lora_config()
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()  # ~0.5% trainable
    return model


def load_finetuned_model(base_model_id: str, lora_weights_path: str, merge: bool = True):
    """Load LoRA weights onto a base model, optionally merging for faster inference."""
    import torch
    from peft import PeftModel
    from transformers import AutoModelForImageTextToText

    base_model = AutoModelForImageTextToText.from_pretrained(
        base_model_id, torch_dtype=torch.bfloat16, device_map="auto"
    )
    model = PeftModel.from_pretrained(base_model, lora_weights_path)
    if merge:
        model = model.merge_and_unload()  # Merge for faster inference
    return model


@dataclass
class EvalResult:
    accuracy: float
    total: int
    correct: int
    details: List[Dict]


def evaluate_video_qa(model, test_data: List[Dict], fuzzy_match: bool = True) -> EvalResult:
    """Compare model predictions against expected answers.

    Each test item is a dict with "video", "question", and "answer" keys.
    With fuzzy_match=True, a prediction counts as correct when the expected
    answer appears anywhere in it (case insensitive).
    """
    correct = 0
    details = []
    for item in test_data:
        prediction = model.ask(item["video"], item["question"])
        expected = item["answer"]
        if fuzzy_match:
            is_correct = expected.lower() in prediction.lower()
        else:
            is_correct = prediction.strip().lower() == expected.strip().lower()
        if is_correct:
            correct += 1
        details.append({"expected": expected, "predicted": prediction, "correct": is_correct})

    return EvalResult(
        correct / len(test_data) if test_data else 0, len(test_data), correct, details)
