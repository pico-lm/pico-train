"""
Utilities for checkpointing evaluation-related states (i.e. evaluation results, etc.)

We save the evaluation results in a JSON file at the step-specific evaluation results directory.
"""

import json
import os
from typing import Any, Dict

from huggingface_hub import upload_folder
from lightning.fabric import Fabric
from lightning.fabric.utilities.rank_zero import rank_zero_only

from src.config import CheckpointingConfig
from src.training.utils.io import use_backoff


@rank_zero_only
@use_backoff()
def save_evaluation_results(
    checkpointing_config: CheckpointingConfig,
    checkpoint_step: int,
    fabric: Fabric,
    evaluation_results: Dict[str, Any],
) -> None:
    """Save evaluation results to disk and optionally to HuggingFace Hub.

    The evaluation results are saved in the following directory structure:
    {checkpointing_config.runs_dir}/
        └── {checkpointing_config.run_name}/
            └── {checkpointing_config.eval_results_dir}/
                └── step_{checkpoint_step}.json

    NOTE: this function is only called on rank 0 to avoid conflicts; assumes that the evaluation
    results are gathered on rank 0.

    Args:
        checkpointing_config: Configuration object containing checkpoint settings
        checkpoint_step: Current training checkpoint step (i.e. number of learning steps taken)
        fabric: Lightning Fabric instance
        evaluation_results: Dictionary containing evaluation metrics
    """

    run_dir = os.path.join(checkpointing_config.runs_dir, checkpointing_config.run_name)
    eval_results_dir = os.path.join(run_dir, checkpointing_config.evaluation.eval_results_dir)

    os.makedirs(eval_results_dir, exist_ok=True)

    curr_eval_results_path = os.path.join(eval_results_dir, f"step_{checkpoint_step}.json")

    # save out as json
    with open(curr_eval_results_path, "w") as f:
        json.dump(evaluation_results, f)

    if checkpointing_config.save_to_hf:
        upload_folder(
            folder_path=eval_results_dir,
            path_in_repo=checkpointing_config.evaluation.eval_results_dir,
            repo_id=checkpointing_config.hf_checkpoint.repo_id,
            commit_message=f"Saving Evaluation Results -- Step {checkpoint_step}",
            revision=checkpointing_config.run_name,
            token=os.getenv("HF_TOKEN"),
        )
