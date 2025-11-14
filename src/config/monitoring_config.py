"""
Monitoring Config

Specifies the monitoring process, e.g. how to log metrics and keep track of training progress.
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class LoggingConfig:
    log_level: str = "INFO"
    log_every_n_steps: int = 100


@dataclass
class WandbConfig:
    # configure logging to Weights and Biases
    project: str = ""
    entity: str = ""


@dataclass
class PicoReportConfig:
    """
    Configuration for Pico Report integration.

    Note: Requires PICO_API_KEY and PICO_LAB_HASH environment variables to be set.
    """

    lab_hash: Optional[str] = None
    experiment_name: Optional[str] = None

    # Git tracking: automatically create git commits for each experiment
    # This captures the exact code state and links it to your experiment in the dashboard
    auto_commit: bool = True


@dataclass
class MonitoringConfig:
    logging: LoggingConfig = field(default_factory=LoggingConfig)

    # Weights and Biases
    save_to_wandb: bool = False
    wandb: WandbConfig = field(default_factory=WandbConfig)

    # Pico Labs - A platform to easily run and share experiments on the web
    # Automatically tracks training metrics, evaluation results, and checkpoints
    # to your private dashboard at https://picolabs.space
    save_to_picolabs: bool = False
    pico_report: PicoReportConfig = field(default_factory=PicoReportConfig)
