"""
Monitoring Config

Specifies the monitoring process, e.g. how to log metrics and keep track of training progress.
"""

from dataclasses import dataclass, field


@dataclass
class LoggingConfig:
    log_level: str = "INFO"
    log_every_n_steps: int = 100


@dataclass
class WandbConfig:
    project: str = ""
    entity: str = ""


@dataclass
class MonitoringConfig:
    logging: LoggingConfig = field(default_factory=LoggingConfig)

    # Weights and Biases
    save_to_wandb: bool = False
    wandb: WandbConfig = field(default_factory=WandbConfig)
