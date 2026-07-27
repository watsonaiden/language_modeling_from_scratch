import tomllib

from typing import Literal
from pydantic import BaseModel


class ModelParams(BaseModel):
    d_model: int
    context_length: int
    num_layers: int
    num_heads: int
    d_ff: int
    rope_theta: int


class TrainingParams(BaseModel):
    lr_max: float
    lr_min: float
    warmup_steps: int
    annealing_steps: int


class TrainingConfig(BaseModel):
    dataset_path: str
    vocab_size: int
    batch_size: int

    # other
    device: Literal["cpu", "cuda", "mps"] = "cpu"
    checkpoint_dir: str | None = None

    model: ModelParams
    training: TrainingParams

    test_mode: bool = False


def get_config(path: str) -> TrainingConfig:

    with open(path, "rb") as fs:
        data = tomllib.load(fs)

    return TrainingConfig.model_validate(data)
