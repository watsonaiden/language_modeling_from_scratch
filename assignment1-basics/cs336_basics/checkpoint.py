import torch


def save_checkpoint(model: torch.nn.Module, optimizer: torch.optim.Optimizer, iteration: int, out):

    data = {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "iteration": iteration}

    torch.save(data, out)


# returns iteration number
def load_checkpoint(src, model: torch.nn.Module, optimizer: torch.optim.Optimizer) -> int:

    data = torch.load(src)

    model.load_state_dict(data["model"])

    optimizer.load_state_dict(data["optimizer"])

    return data["iteration"]
