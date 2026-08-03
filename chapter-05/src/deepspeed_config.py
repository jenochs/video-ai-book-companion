"""DeepSpeed configuration for supervised fine-tuning (SFT).

Companion module for Chapter 5, "Fine-Tuning for Specific Video Tasks",
from the book "Video Generation with AI". Provides the production ZeRO-2
configuration used for full supervised fine-tuning of CogVideoX. ZeRO
partitions optimizer states and gradients across devices, and CPU
offloading pushes optimizer memory to system RAM, which lets a 5B
parameter model train on a single high-memory GPU.
"""


class DeepSpeedConfig:
    """Production DeepSpeed configurations for different hardware."""

    @staticmethod
    def get_zero2_config():
        """ZeRO-2: For single GPU with 36GB+ VRAM"""
        return {
            "train_batch_size": "auto",
            "gradient_accumulation_steps": "auto",
            "gradient_clipping": 1.0,
            "zero_optimization": {
                "stage": 2,
                "offload_optimizer": {
                    "device": "cpu",  # Offload to CPU
                    "pin_memory": True
                },
                "allgather_partitions": True,
                "allgather_bucket_size": 2e8,
                "reduce_scatter": True,
                "reduce_bucket_size": 2e8,
                "overlap_comm": True,
                "contiguous_gradients": True
            },
            "fp16": {
                "enabled": True,
                "loss_scale": 0,
                "loss_scale_window": 1000,
                "initial_scale_power": 16,
                "hysteresis": 2,
                "min_loss_scale": 1
            }
        }
