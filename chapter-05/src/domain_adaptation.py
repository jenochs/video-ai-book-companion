"""Domain adaptation framework for video models.

Companion module for Chapter 5, "Fine-Tuning for Specific Video Tasks",
from the book "Video Generation with AI". Implements adversarial domain
adaptation: a gradient reversal layer, a domain discriminator, a video
model wrapper with domain-specific batch normalization and domain
embeddings, and a trainer that balances task loss on the source domain
against domain confusion, so the model learns representations that
transfer to targets like medical imaging or satellite footage.
"""

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Function
from torch.utils.data import DataLoader


class GradientReversalFunction(Function):
    """Gradient reversal for adversarial domain adaptation"""

    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.lambda_, None


class DomainDiscriminator(nn.Module):
    """Discriminator for adversarial domain adaptation"""

    def __init__(self, feature_dim: int, hidden_dim: int = 256):
        super().__init__()

        self.discriminator = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, x: torch.Tensor, lambda_: float = 1.0):
        # Apply gradient reversal
        x = GradientReversalFunction.apply(x, lambda_)
        return self.discriminator(x)


class DomainAdaptationVideoModel(nn.Module):
    """Video model with domain adaptation capabilities"""

    def __init__(self,
                 base_model: nn.Module,
                 feature_dim: int,
                 num_domains: int = 2,
                 adaptation_method: str = 'adversarial'):
        super().__init__()

        self.base_model = base_model
        self.feature_dim = feature_dim
        self.num_domains = num_domains
        self.adaptation_method = adaptation_method

        # Domain discriminator for adversarial adaptation
        if adaptation_method == 'adversarial':
            self.domain_discriminator = DomainDiscriminator(feature_dim)

        # Domain-specific batch norm layers
        self.domain_bn = nn.ModuleList([
            nn.BatchNorm1d(feature_dim) for _ in range(num_domains)
        ])

        # Feature alignment layers
        self.alignment_proj = nn.Linear(feature_dim, feature_dim)

        # Domain embedding for conditioning
        self.domain_embed = nn.Embedding(num_domains, feature_dim)

    def extract_features(self, x: torch.Tensor, domain_id: int = None):
        """Extract features with optional domain-specific normalization"""
        features = self.base_model.encode(x)

        # Apply domain-specific batch normalization
        if domain_id is not None and domain_id < self.num_domains:
            B, T, D = features.shape
            features = features.reshape(B * T, D)
            features = self.domain_bn[domain_id](features)
            features = features.reshape(B, T, D)

        return features

    def forward(self, x: torch.Tensor, domain_id: int = None,
                return_domain_loss: bool = False, lambda_: float = 1.0):
        # Extract features
        features = self.extract_features(x, domain_id)

        # Apply domain conditioning if available
        if domain_id is not None:
            domain_emb = self.domain_embed(torch.tensor([domain_id], device=x.device))
            features = features + domain_emb.unsqueeze(1)

        # Generate video
        output = self.base_model.decode(features)

        if return_domain_loss and self.adaptation_method == 'adversarial':
            # Compute domain classification loss
            domain_pred = self.domain_discriminator(features.mean(dim=1), lambda_)
            return output, domain_pred

        return output


class DomainAdaptationTrainer:
    """Trainer for domain adaptation of video models"""

    def __init__(self,
                 model: DomainAdaptationVideoModel,
                 source_loader: DataLoader,
                 target_loader: DataLoader,
                 optimizer: torch.optim.Optimizer,
                 device: str = 'cuda'):

        self.model = model
        self.source_loader = source_loader
        self.target_loader = target_loader
        self.optimizer = optimizer
        self.device = device

        # Loss functions
        self.task_criterion = nn.MSELoss()
        self.domain_criterion = nn.BCEWithLogitsLoss()

        # Training statistics
        self.global_step = 0
        self.epoch = 0

    def train_epoch(self, lambda_schedule: callable = None):
        """Train for one epoch with domain adaptation"""
        self.model.train()

        source_iter = iter(self.source_loader)
        target_iter = iter(self.target_loader)

        total_task_loss = 0
        total_domain_loss = 0

        for step in range(len(self.source_loader)):
            # Get lambda for gradient reversal
            if lambda_schedule:
                lambda_ = lambda_schedule(self.global_step)
            else:
                lambda_ = 2 / (1 + np.exp(-10 * self.global_step / 1000)) - 1

            # Load source batch
            try:
                source_batch = next(source_iter)
            except StopIteration:
                source_iter = iter(self.source_loader)
                source_batch = next(source_iter)

            # Load target batch
            try:
                target_batch = next(target_iter)
            except StopIteration:
                target_iter = iter(self.target_loader)
                target_batch = next(target_iter)

            # Move to device
            source_videos = source_batch['frames'].to(self.device)
            target_videos = target_batch['frames'].to(self.device)

            # Forward pass for source domain (with task loss)
            source_output, source_domain_pred = self.model(
                source_videos, domain_id=0,
                return_domain_loss=True, lambda_=lambda_
            )

            # Forward pass for target domain (no task loss)
            _, target_domain_pred = self.model(
                target_videos, domain_id=1,
                return_domain_loss=True, lambda_=lambda_
            )

            # Task loss (only for source domain)
            task_loss = self.task_criterion(source_output, source_videos)

            # Domain classification loss
            source_domain_labels = torch.zeros(
                source_domain_pred.shape[0], 1
            ).to(self.device)
            target_domain_labels = torch.ones(
                target_domain_pred.shape[0], 1
            ).to(self.device)

            domain_loss = self.domain_criterion(
                source_domain_pred, source_domain_labels
            ) + self.domain_criterion(
                target_domain_pred, target_domain_labels
            )

            # Combined loss
            total_loss = task_loss + 0.1 * domain_loss

            # Backward pass
            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()

            # Update statistics
            total_task_loss += task_loss.item()
            total_domain_loss += domain_loss.item()
            self.global_step += 1

            if step % 100 == 0:
                print(f"Step {step}: Task Loss: {task_loss:.4f}, "
                      f"Domain Loss: {domain_loss:.4f}, Lambda: {lambda_:.3f}")

        self.epoch += 1
        return {
            'task_loss': total_task_loss / len(self.source_loader),
            'domain_loss': total_domain_loss / len(self.source_loader)
        }
