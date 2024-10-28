import click
import logging
import lightning as L
import torch
import torch.nn.functional as F
from lightning.fabric.utilities.rank_zero import rank_zero_only

from model import Pico

from lightning.pytorch.demos import WikiText2
from torch.utils.data import DataLoader

from utils.initialization import (
    initialize_run_dir, 
    initialize_fabric, 
    initialize_config, 
    initialize_logging,
    initialize_optimizer, 
    initialize_lr_scheduler, 
    initialize_checkpointing
)
from utils.checkpointing import (
    load_checkpoint, 
    save_checkpoint,
    save_config
)

from utils import third_party_auth_checks

@click.command()
@click.option("--config_path", type=str,)
def main(config_path: str):
    """
    Core Training Loop. 
    """

    ########################################################
    #
    # Pre-Training Setup (aka. boilerplate)
    #
    ########################################################

    # ---- Setup Configs ---- #
    data_config, model_config, training_config, evaluation_config = initialize_config(config_path)

    # Check that the user has access to third-party services (HuggingFace, Weights & Biases)
    third_party_auth_checks(training_config)

    # ---- Setup Run Directory ---- #
    initialize_run_dir(training_config)

    # ---- Setup Logger ---- #
    logger, experiment_tracker = initialize_logging(training_config)
    @rank_zero_only
    def log(msg, level=logging.INFO): logger.log(level, msg)

    # ---- Setup Fabric ---- #
    fabric = initialize_fabric(training_config, experiment_tracker)

    # ---- Setup Dataset, Tokenizer, and Dataloader ---- #
    dataset = WikiText2()
    train_dataloader = DataLoader(dataset, batch_size=model_config.batch_size, shuffle=True)
    model_config.vocab_size = dataset.vocab_size

    # ---- Setup Model, Optimizer, and Dataloaders ---- #
    model = Pico(model_config, fabric)
    optimizer = initialize_optimizer(model, training_config)
    lr_scheduler = initialize_lr_scheduler(optimizer, training_config)

    # ---- Wrapping with Fabric ---- #
    model, optimizer = fabric.setup(model, optimizer)
    train_dataloader = fabric.setup_dataloaders(train_dataloader)

    # ---- Load Checkpoint (if specified) ---- #
    initialize_checkpointing(training_config)
    L.seed_everything(42)

    should_load_checkpoint = training_config.checkpointing.load_checkpoint_path is not None or training_config.checkpointing.load_latest_checkpoint
    if should_load_checkpoint:
        model, optimizer, lr_scheduler, train_start_step, train_iterator = load_checkpoint(
            fabric, training_config, model, optimizer, lr_scheduler, train_dataloader
        )
    else:
        train_start_step = 0
        train_iterator = iter(train_dataloader)
        save_config(fabric, training_config, model_config, evaluation_config)
        save_checkpoint(fabric, training_config, model, optimizer, lr_scheduler, 0)

    ########################################################
    #
    # Training Configs
    #
    ########################################################

    if train_start_step < training_config.training_steps:
        log(f"Training from step {train_start_step}")

    gradient_step = train_start_step
    # Loss tracking over log_every_n_steps interval 
    interval_loss = torch.tensor(0.0, device=fabric.device)
    interval_steps = torch.tensor(0, device=fabric.device)
    interval_inf_or_nan_count = torch.tensor(0, device=fabric.device)

    for batch_idx, batch in enumerate(train_iterator, start=train_start_step):
        input_ids, labels = batch
        model_output = model(input_ids).transpose(1, 2)

        # ---- Gradient Accumulation (if enabled) ---- #
        should_accumulate_gradients = (batch_idx+1) % training_config.optimization.gradient_accumulation_steps != 0

        with fabric.no_backward_sync(model, enabled=should_accumulate_gradients):
            # Compute loss
            loss = F.cross_entropy(model_output, labels)
            fabric.backward(loss/training_config.optimization.gradient_accumulation_steps)

            if torch.isnan(loss) or torch.isinf(loss):
                interval_inf_or_nan_count += 1
            else:
                interval_loss += loss.item()
                interval_steps += 1

        if should_accumulate_gradients:
            continue

        # NOTE: Code after this point only runs after gradient_accumulation_steps
    
        # ---- Logging ---- #

        if gradient_step % training_config.logging.log_every_n_steps == 0:
            gathered_interval_loss = fabric.all_reduce(interval_loss, reduce_op="sum").item()
            gathered_interval_inf_or_nan_count = fabric.all_reduce(interval_inf_or_nan_count, reduce_op="sum").item()
            gathered_interval_steps = fabric.all_reduce(interval_steps, reduce_op="sum").item()
            
            if gathered_interval_steps > 0:
                avg_loss = gathered_interval_loss / gathered_interval_steps
            else:
                avg_loss = float('inf')  # or some other appropriate value

            interval_loss = torch.tensor(0.0, device=fabric.device)
            interval_steps = torch.tensor(0, device=fabric.device)
            interval_inf_or_nan_count = torch.tensor(0, device=fabric.device)

            fabric.log("loss", avg_loss, step=gradient_step)
            fabric.log("inf_or_nan_count", gathered_interval_inf_or_nan_count, step=gradient_step)
            fabric.log("learning_rate", lr_scheduler.get_last_lr()[0], step=gradient_step)
            log(f"Step {gradient_step} Loss: {avg_loss}, Inf/NaN count: {gathered_interval_inf_or_nan_count}")

        # ---- Gradient Step ---- #

        fabric.clip_gradients(model, optimizer, max_norm=training_config.optimization.max_norm)
        optimizer.step()
        optimizer.zero_grad()
        lr_scheduler.step()

        gradient_step += 1

        # ---- Checkpointing ---- #
        # Maybe save checkpoint 
        if gradient_step % training_config.checkpointing.save_every_n_steps == 0:
            log(f"Saving checkpoint at step {gradient_step}")
            save_checkpoint(fabric, training_config, model, optimizer, lr_scheduler, gradient_step)

        # --- Evaluation --- #
        if gradient_step % evaluation_config.eval_every_n_steps == 0:
            log("Starting Evaluation!")

        # --- Break Training Condition --- #
        if gradient_step == training_config.training_steps:
            # Save final checkpoint if we didn't save it at already 
            if gradient_step % training_config.checkpointing.save_every_n_steps != 0:
                log(f"Saving final checkpoint at step {gradient_step}")
                save_checkpoint(fabric, training_config, model, optimizer, lr_scheduler, gradient_step)
            break

    # --- Final Evaluation --- #
    log("Starting Final Evaluation!")

if __name__ == "__main__":
    main()
