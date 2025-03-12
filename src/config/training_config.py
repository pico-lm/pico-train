"""
Training Config

Specifies the hyperparameters for the training process, i.e. the optimizer, learning rate, etc.
"""

from dataclasses import dataclass, field
from ._constants import GRADIENT_ACCUMULATION_STEPS


@dataclass
class FabricConfig:
    # Configure nodes/devices for parallelised training
    num_nodes: int = 1
    num_devices: int = 1
    precision: str = "bf16-mixed"
    # Hardware accelerator to use, can be cpu/cuda/mps etc.
    accelerator: str = "cuda"


@dataclass
class OptimizationConfig:
    # Optimizer
    optimizer: str = "adamw"
    lr: float = 3e-4

    # Learning Rate Scheduler
    lr_scheduler: str = "linear_with_warmup"
    lr_warmup_steps: int = 2500

    # Define number of gradient accumulation steps
    gradient_accumulation_steps: int = GRADIENT_ACCUMULATION_STEPS


@dataclass
class TrainingConfig:
    fabric: FabricConfig = field(default_factory=FabricConfig)
    optimization: OptimizationConfig = field(default_factory=OptimizationConfig)
    max_steps: int = 200_000
