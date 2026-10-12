import json
from pathlib import Path
from typing import Optional

import torch

from cs336_basics.config import get_config
from cs336_basics.modules.transformer_lm import TransformerLM


def save_checkpoint(
    model: TransformerLM, optimizer: torch.optim.Optimizer, iterations: list[int], losses: dict[str, list[float]], out
):
    # operation fails if parent dir doesn't exist
    dir = Path(out)
    dir.mkdir(parents=True, exist_ok=True)

    state = {
        "iter": max(iterations),
        "lm_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
    }

    torch.save(state, dir / "model_checkpoint.pth")

    (dir / "loss_data.json").write_text(json.dumps({"iterations": iterations, **losses}))


# returns iteration number
def load_checkpoint(
    results_dir: Path, model: Optional[TransformerLM], optimizer: Optional[torch.optim.Optimizer]
) -> int:
    state_path = results_dir / "model_checkpoint.pth"

    if not state_path.exists():
        print(f"{state_path} does not exist yet skipping loading")
        return 0

    print(f"Loading state from {state_path}")

    state = torch.load(state_path)

    if model:
        model.load_state_dict(state["lm_state"])
    if optimizer:
        optimizer.load_state_dict(state["optimizer_state"])

    return state["iter"] if "iter" in state else 0


def get_checkpoint_model(config_path) -> TransformerLM:
    print(f"loading config from {config_path}")
    cfg = get_config(config_path)

    print(f"initializing model with {cfg.model}")
    lm = TransformerLM(vocab_size=cfg.vocab_size, **cfg.model.model_dump())

    load_checkpoint(Path(cfg.results_dir), lm, None)

    return lm
