import torch
import numpy as np
import click
from tqdm.auto import tqdm

from cs336_basics.modules import TransformerLM
from cs336_basics.config import get_config, TrainingConfig
from cs336_basics.data import get_batch
from cs336_basics.training import AdamW, cross_entropy_loss, lr_cosine_schedule, gradient_clipping


@torch.no_grad()
def run_validation(data: np.ndarray, model: TransformerLM, cfg: TrainingConfig):
    losses = []

    for _ in range(cfg.validation.num_batches):
        batch_input, batch_exp_output = get_batch(
            data, batch_size=cfg.batch_size, context_length=cfg.model.context_length, device=cfg.device
        )

        output = model(batch_input)

        # squeeze batches together so we have
        # token_i_pred, logits
        # token_i_real
        loss = cross_entropy_loss(output.view(-1, output.size(-1)), batch_exp_output.view(-1))

        losses.append(loss.item())

    return np.mean(losses)


@click.command()
@click.option("--config-path", required=True)
def train(config_path: str):

    print(f"loading config from {config_path}")
    cfg = get_config(config_path)
    print(f"running training on {cfg.dataset_path} ({cfg.vocab_size} unique tokens)")

    print(f"initializing model with {cfg.model}")
    lm = TransformerLM(vocab_size=cfg.vocab_size, **cfg.model.model_dump())

    print(f"running on {cfg.device}")
    lm.to(cfg.device)

    optimizer = AdamW(lm.parameters(), lr=cfg.training.lr_max)
    data = np.memmap(cfg.dataset_path, dtype=np.uint16, mode="r")

    if cfg.validation.validation_path:
        data_validation = np.memmap(cfg.validation.validation_path, dtype=np.uint16, mode="r")
        print(f"adding validation set from {cfg.validation.validation_path}")
    else:
        print("validation set not configured")

    learning_steps = cfg.training.annealing_steps + cfg.training.warmup_steps
    print(f"estimated training tokens {cfg.model.context_length * learning_steps * cfg.batch_size:,}")

    if cfg.test_mode:
        print("In Testing Mode using only one mini-batch")
        batch_input, batch_exp_output = get_batch(
            data, batch_size=cfg.batch_size, context_length=cfg.model.context_length, device=cfg.device
        )

    for iter in tqdm(range(learning_steps)):
        # only regenerate if not in test mode
        if not cfg.test_mode:
            batch_input, batch_exp_output = get_batch(
                data, batch_size=cfg.batch_size, context_length=cfg.model.context_length, device=cfg.device
            )

        optimizer.zero_grad()

        output = lm(batch_input)

        # squeeze batches together so we have
        # token_i_pred, logits
        # token_i_real
        loss = cross_entropy_loss(output.view(-1, output.size(-1)), batch_exp_output.view(-1))
        loss.backward()

        # validation
        if iter % cfg.validation.validation_step_rate == 0:
            if data_validation is not None:
                tqdm.write(f"iter: {iter}: validation loss {run_validation(data_validation, lm, cfg)}")
            tqdm.write(f"iter {iter}: training loss {loss}")

        gradient_clipping(lm.parameters(), 1)
        lr_cosine_schedule(iter, **cfg.training.model_dump())
        optimizer.step()

    if cfg.checkpoint_dir:
        print("writing to checkpoint", cfg.checkpoint_dir)


if __name__ == "__main__":
    train()
