"""Training infrastructure for video generation models.

Accompanies Chapter 4 of "Video Generation with AI" (O'Reilly) by Joseph Enochs.

Implements the "Training Infrastructure and Optimization" section: a trainer
with AdamW and selective weight decay, warmup plus cosine scheduling, gradient
accumulation, optional mixed precision, EMA weights, checkpointing, and
distributed (DDP) support.
"""

import copy
import os
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel as DDP


def enable_tf32():
    """Enable TF32 as in Latte's train.py.

    These two lines give roughly 2x speedup on A100/H100 GPUs.
    Note: Latte disables these for FP16 training, so set them back to False
    if you switch to mixed precision.
    """
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


def setup_distributed():
    """Minimal distributed setup following Latte (run under torchrun)."""
    dist.init_process_group("nccl")
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    return rank, local_rank, device


class VideoModelTrainer:
    """
    Educational trainer for video generation models.
    Handles distributed training, memory optimization, and checkpointing.
    """

    def __init__(self,
                 model: nn.Module,
                 train_dataloader,
                 val_dataloader=None,
                 learning_rate: float = 1e-4,
                 warmup_steps: int = 10000,
                 total_steps: int = 1000000,
                 num_timesteps: int = 1000,
                 gradient_clip: float = 1.0,
                 ema_decay: float = 0.9999,
                 mixed_precision: bool = True,
                 gradient_accumulation_steps: int = 1,
                 checkpoint_dir: str = "./checkpoints",
                 distributed: bool = False):

        # Core components
        self.model = model
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader

        # Training configuration
        self.gradient_clip = gradient_clip
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.mixed_precision = mixed_precision
        self.distributed = distributed
        self.num_timesteps = num_timesteps
        self.ema_decay = ema_decay

        # Setup device and distributed training
        if distributed:
            self.setup_distributed()
        else:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.model = self.model.to(self.device)

        # Initialize training components
        self.optimizer = self._build_optimizer(learning_rate)
        self.scheduler = self._build_scheduler(warmup_steps, total_steps)
        self.scaler = GradScaler() if mixed_precision else None
        self.ema_model = self._build_ema_model(ema_decay) if ema_decay else None

        # Training state tracking
        self.global_step = 0
        self.epoch = 0
        self.best_val_loss = float('inf')

        # Checkpointing setup
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(exist_ok=True, parents=True)

    # Part 2: Distributed training configuration
    def setup_distributed(self):
        """Initialize distributed training across multiple GPUs."""
        # Initialize process group for inter-GPU communication
        dist.init_process_group(backend='nccl')  # NCCL optimized for NVIDIA GPUs

        self.rank = dist.get_rank()  # This GPU's ID
        self.world_size = dist.get_world_size()  # Total number of GPUs
        self.device = torch.device(f'cuda:{self.rank}')

        # Ensure each process uses its assigned GPU
        torch.cuda.set_device(self.device)
        self.model = self.model.to(self.device)

        # Wrap model for distributed training
        self.model = DDP(self.model, device_ids=[self.rank])

    # Part 3: Optimizer and scheduler configuration
    def _build_optimizer(self, learning_rate):
        """Build AdamW with selective weight decay."""
        # Separate parameters by weight decay requirement
        decay_params = []
        no_decay_params = []

        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue

            # Biases and norm parameters should not be regularized
            if 'bias' in name or 'norm' in name:
                no_decay_params.append(param)
            else:
                decay_params.append(param)

        # AdamW with different weight decay for different params
        optimizer = torch.optim.AdamW([
            {'params': decay_params, 'weight_decay': 0.01},
            {'params': no_decay_params, 'weight_decay': 0.0}
        ], lr=learning_rate, betas=(0.9, 0.999), eps=1e-8)

        print(f"Optimizer: {len(decay_params)} params with decay, "
              f"{len(no_decay_params)} params without decay")

        return optimizer

    def _build_scheduler(self, warmup_steps, total_steps=1000000):
        """Cosine annealing with linear warmup."""
        warmup_steps = max(warmup_steps, 1)

        def lr_lambda(step):
            if step < warmup_steps:
                # Linear warmup
                return step / warmup_steps
            # Cosine annealing after warmup
            progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
            return 0.5 * (1 + np.cos(np.pi * progress))

        return torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)

    def _build_ema_model(self, decay):
        """Initialize EMA model for stable generation."""
        ema_model = copy.deepcopy(self.model)
        ema_model.eval()
        ema_model.requires_grad_(False)
        return ema_model

    # Part 4: Gradient accumulation and memory management
    @contextmanager
    def _gradient_accumulation_context(self, step_idx):
        """
        Control gradient synchronization in distributed training.
        Key insight: only communicate gradients when ready to update.
        """
        if self.distributed:
            if step_idx < self.gradient_accumulation_steps - 1:
                # Accumulate locally - no inter-GPU communication
                with self.model.no_sync():
                    yield
            else:
                # Final accumulation - sync gradients across GPUs
                yield
        else:
            yield

    def _split_batch(self, batch, num_splits):
        """Split batch into micro-batches for accumulation."""
        batch_size = len(batch['frames'])
        micro_size = max(batch_size // num_splits, 1)

        micro_batches = []
        for i in range(num_splits):
            start = i * micro_size
            if start >= batch_size:
                break
            end = start + micro_size if i < num_splits - 1 else batch_size
            micro_batches.append({
                'frames': batch['frames'][start:end],
                'captions': batch['captions'][start:end]
            })
        return micro_batches

    # Part 5: Core training step implementation
    def train_step(self, batch):
        """Execute one complete training step."""
        total_loss = 0

        # Split batch for gradient accumulation
        micro_batches = self._split_batch(batch, self.gradient_accumulation_steps)

        # Process each micro-batch
        for i, micro_batch in enumerate(micro_batches):
            with self._gradient_accumulation_context(i):
                if self.mixed_precision:
                    # FP16 computation with automatic scaling
                    with autocast():
                        loss = self._compute_loss(micro_batch)

                    # Scale loss by accumulation steps
                    loss = loss / self.gradient_accumulation_steps

                    # Backward with gradient scaling for FP16
                    self.scaler.scale(loss).backward()
                else:
                    # Standard FP32 computation
                    loss = self._compute_loss(micro_batch)
                    loss = loss / self.gradient_accumulation_steps
                    loss.backward()

                total_loss += loss.item()

        # Gradient clipping prevents exploding gradients
        if self.mixed_precision:
            # Unscale gradients before clipping
            self.scaler.unscale_(self.optimizer)

        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.model.parameters(),
            self.gradient_clip
        )

        # Optimizer step with mixed precision handling
        if self.mixed_precision:
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            self.optimizer.step()

        # Clear gradients (set_to_none saves memory)
        self.optimizer.zero_grad(set_to_none=True)
        self.scheduler.step()

        # Update EMA model if configured
        if self.ema_model:
            self._update_ema()

        self.global_step += 1

        return {
            'loss': total_loss,
            'grad_norm': grad_norm.item(),
            'learning_rate': self.scheduler.get_last_lr()[0]
        }

    # Part 6: Loss computation and model updates
    def _compute_loss(self, batch):
        """
        Compute diffusion loss for video generation.
        Trains the model to predict noise added at various timesteps.
        """
        frames = batch['frames'].to(self.device)
        captions = batch['captions']

        # Sample random timesteps for each video
        batch_size = frames.shape[0]
        timesteps = torch.randint(
            0, self.num_timesteps,  # Diffusion timesteps
            (batch_size,),
            device=self.device
        )

        # Generate noise and create noisy videos
        noise = torch.randn_like(frames)

        # Forward diffusion: add noise according to schedule
        noisy_frames = self.add_noise(frames, noise, timesteps)

        # Model predicts the noise that was added
        predicted_noise = self.model(
            noisy_frames,
            timesteps,
            captions
        )

        # MSE loss between predicted and actual noise
        loss = torch.nn.functional.mse_loss(predicted_noise, noise)

        return loss

    def add_noise(self, x, noise, t):
        """Add noise to frames using diffusion schedule."""
        # Simplified linear schedule (replace with cosine for production)
        alpha = (1 - t.float() / self.num_timesteps).view(-1, 1, 1, 1, 1)
        return alpha * x + (1 - alpha).sqrt() * noise

    def _update_ema(self):
        """
        Update exponential moving average model.
        EMA provides stable weights for generation.
        """
        with torch.no_grad():
            decay = self.ema_decay
            for ema_param, model_param in zip(
                self.ema_model.parameters(),
                self.model.parameters()
            ):
                # Weighted average: mostly old weights, small contribution from new
                ema_param.data = decay * ema_param.data + (1 - decay) * model_param.data

    # Part 7: Training loop and monitoring
    def train_epoch(self):
        """Complete one training epoch with monitoring."""
        self.model.train()
        epoch_metrics = []

        for batch_idx, batch in enumerate(self.train_dataloader):
            # Execute training step
            metrics = self.train_step(batch)
            epoch_metrics.append(metrics)

            # Periodic logging (every 100 steps)
            if self.global_step % 100 == 0:
                print(f"Step {self.global_step}: Loss={metrics['loss']:.4f}, "
                      f"Grad={metrics['grad_norm']:.4f}, LR={metrics['learning_rate']:.2e}")

            # Periodic validation (every 1000 steps)
            if self.global_step % 1000 == 0 and self.val_dataloader:
                val_metrics = self.validate()
                if val_metrics['loss'] < self.best_val_loss:
                    self.best_val_loss = val_metrics['loss']
                    self.save_checkpoint('best')

            # Regular checkpointing (every 5000 steps)
            if self.global_step % 5000 == 0:
                self.save_checkpoint(f'step_{self.global_step}')

        self.epoch += 1
        return epoch_metrics

    def validate(self):
        """Run validation and return metrics."""
        self.model.eval()
        val_losses = []

        with torch.no_grad():
            for batch in self.val_dataloader:
                loss = self._compute_loss(batch)
                val_losses.append(loss.item())

        self.model.train()
        return {'loss': np.mean(val_losses)}

    def save_checkpoint(self, name):
        """Save training state for resume."""
        checkpoint = {
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'ema_model': self.ema_model.state_dict() if self.ema_model else None,
            'scaler': self.scaler.state_dict() if self.scaler else None,
            'global_step': self.global_step,
            'epoch': self.epoch,
            'best_val_loss': self.best_val_loss
        }
        torch.save(checkpoint, self.checkpoint_dir / f'{name}.pt')
