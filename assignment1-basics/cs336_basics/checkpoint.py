from pathlib import Path
from typing import Optional

import torch

from cs336_basics.config import get_config
from cs336_basics.modules.transformer_lm import TransformerLM


def save_checkpoint(model: TransformerLM, optimizer: torch.optim.Optimizer, iteration: int, out):
    # operation fails if parent dir doesn't exist
    parent_dir = Path(out).parent
    parent_dir.mkdir(parents=True, exist_ok=True)

    state = {
        "iter": iteration,
        "lm_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
    }

    torch.save(state, out)


# returns iteration number
def load_checkpoint(src, model: Optional[TransformerLM], optimizer: Optional[torch.optim.Optimizer]) -> int:
    state = torch.load(src)

    if model:
        model.load_state_dict(state["lm_state"])
    if optimizer:
        optimizer.load_state_dict(state["optimizer_state"])

    return state["iter"] if 'iter' in state else 0


def get_checkpoint_model(config_path) -> TransformerLM:
    print(f"loading config from {config_path}")
    cfg = get_config(config_path)

    print(f"initializing model with {cfg.model}")
    lm = TransformerLM(vocab_size=cfg.vocab_size, **cfg.model.model_dump())

    load_checkpoint(cfg.checkpoint_path, lm, None)

    return lm