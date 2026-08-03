"""LoRA configuration for CogVideoX fine-tuning.

Companion module for Chapter 5, "Fine-Tuning for Specific Video Tasks",
from the book "Video Generation with AI". Contains the production LoRA
configuration, the helper that applies adapters to the transformer, and
the per-model training configurations discussed in the chapter.

LoRA decomposes weight updates into two low-rank matrices: instead of
updating a full weight matrix W, it learns B and A where W' = W + BA.
Applied to the query, key, value, and output projections of the
CogVideoX transformer, this cuts trainable parameters by over 90%
while keeping most of the adaptation quality.
"""

import torch.nn as nn
from diffusers import CogVideoXTransformer3DModel
from peft import LoraConfig, get_peft_model


class CogVideoLoRAConfig:
    """
    Production LoRA configuration used by CogVideo for different model sizes.
    These settings are proven to work across thousands of fine-tuning runs.
    """

    @staticmethod
    def get_lora_config(model_size: str = "2b", rank: int = 128):
        """Get LoRA configuration for CogVideo models.

        Args:
            model_size: "2b" for 2B model, "5b" for 5B model
            rank: LoRA rank (16 for fast, 128 for quality)

        Returns:
            LoraConfig object for PEFT library
        """

        # Target modules based on CogVideo's implementation
        target_modules = [
            "to_q",  # Query projection
            "to_k",  # Key projection
            "to_v",  # Value projection
            "to_out.0",  # Output projection
        ]

        # Configuration based on model size
        # Note: Repo default lora_alpha=64 (finetune/schemas/args.py:76)
        if model_size == "2b":
            # 2B model: Works on RTX 4080 (16GB)
            config = LoraConfig(
                r=rank,  # LoRA rank (default: 128)
                lora_alpha=64,  # Repo default: 64
                init_lora_weights=True,
                target_modules=target_modules,
            )
        elif model_size == "5b":
            # 5B model: Requires RTX 4090 (24GB)
            config = LoraConfig(
                r=rank,
                lora_alpha=64,  # Repo default: 64
                init_lora_weights=True,
                target_modules=target_modules,
            )
        else:
            raise ValueError(f"Unsupported model size: {model_size}")

        return config


def apply_lora_to_model(transformer: CogVideoXTransformer3DModel,
                        lora_config: LoraConfig) -> nn.Module:
    """Apply LoRA to CogVideo transformer model.

    Args:
        transformer: Base CogVideoX transformer
        lora_config: LoRA configuration

    Returns:
        Model with LoRA adapters applied
    """
    # Apply PEFT (Parameter-Efficient Fine-Tuning) to model
    lora_model = get_peft_model(transformer, lora_config)

    # Print trainable parameters (typically 1-10% of total)
    lora_model.print_trainable_parameters()

    return lora_model


class CogVideoTrainingConfig:
    """Training configuration used by CogVideo."""

    # Model configurations
    model_configs = {
        "cogvideox-2b": {
            "learning_rate": 1e-4,
            "batch_size": 1,  # Per GPU
            "gradient_accumulation": 4,
            "mixed_precision": "fp16",
            "max_grad_norm": 1.0,
            "num_frames": 49,  # Must be 8N+1
            "height": 480,
            "width": 720,
        },
        "cogvideox-5b": {
            "learning_rate": 5e-5,
            "batch_size": 1,
            "gradient_accumulation": 8,
            "mixed_precision": "bf16",
            "max_grad_norm": 1.0,
            "num_frames": 49,
            "height": 480,
            "width": 720,
        },
        "cogvideox1.5-5b": {
            "learning_rate": 5e-5,
            "batch_size": 1,
            "gradient_accumulation": 16,
            "mixed_precision": "bf16",
            "max_grad_norm": 0.5,
            "num_frames": 81,  # Higher frame count
            "height": 768,
            "width": 1360,
        },
    }
