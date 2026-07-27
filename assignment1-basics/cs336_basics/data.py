import torch

import numpy.typing as npt


def get_batch(dataset: npt.NDArray, batch_size: int, context_length: int, device="cpu"):
    indices = torch.randint(0, len(dataset) - context_length, size=(batch_size, 1)) + torch.arange(context_length)

    # return as int as these are used to index into embeddings or in cross entropy which requires int
    return (
        torch.from_numpy(dataset[indices]).to(device).int(),
        torch.from_numpy(dataset[indices + 1]).to(device).int(),
    )
