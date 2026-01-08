import os
import torch
import numpy as np
from typing import Any
from typing import Callable
from argparse import ArgumentParser
from matplotlib import pyplot as plt
plt.style.use('dark_background')

from torch.utils.data import DataLoader
from torch.distributed import init_process_group, destroy_process_group

def normalize_to_neg_one_to_one(img: torch.Tensor) -> torch.Tensor:
    return img * 2 - 1

def unnormalize_to_zero_to_one(img: torch.Tensor) -> torch.Tensor:
    return (img + 1) * 0.5

def dynamic_range_opt(array, epsilon=1e-6, mult_factor=1):
    array = (array + epsilon)/epsilon
    a = np.log10(array)
    b = np.log10(1/epsilon)
    return a/b * mult_factor

def identity(x: torch.Tensor) -> torch.Tensor:
    return x

def default(val: Any, default_val: Any) -> Any:
    return val if val is not None else default_val

def tensor_to_numpy(tensor: torch.Tensor) -> np.ndarray:
    if tensor.is_floating_point():
        return tensor.detach().cpu().numpy()
    else:
        return tensor.detach().cpu().numpy().astype(np.float32)

def extract(
    constants: torch.Tensor, timestamps: torch.Tensor, shape: int
) -> torch.Tensor:
    batch_size = timestamps.shape[0]
    out = constants.gather(-1, timestamps)
    return out.reshape(batch_size, *((1,) * (len(shape) - 1))).to(timestamps.device)

def cycle(dl: DataLoader):
    while True:
        for data in dl:
            yield data

def save_img_grid(
        img_batch:np.ndarray, masses:np.ndarray, model_name:str, milestone:int, cbar=False
        ):
    img_batch  = np.squeeze(img_batch)
    mass_batch = np.log10(np.squeeze(masses)) + 13.8
    batch_size = img_batch.shape[0]
    grid_size  = int(np.sqrt(batch_size))

    fig, axes = plt.subplots(grid_size, grid_size, figsize=(12, 12))
    for i in range(batch_size):
        ax = axes[i // grid_size, i % grid_size]
        img = ax.imshow(img_batch[i], cmap='inferno')
        ax.set_title(f"{mass_batch[i]:.2f}")
        ax.axis('off')
        if cbar:
            # Aggiungi una colorbar accanto a ciascun subplot
            fig.colorbar(img, ax=ax, fraction=0.046, pad=0.04)
            plt.savefig(f'results/{model_name}/mock_grid_cbar_{milestone}.png', bbox_inches='tight', transparent=True)

        else: plt.savefig(f'results/{model_name}/mock_grid_{milestone}.png', bbox_inches='tight', transparent=True)
    plt.close()

def parser(
    prog_name: str, dscr: str, get_args: Callable[[ArgumentParser], ArgumentParser]
) -> Callable[[Callable], Callable]:
    def decorator(function):
        def new_function(*args, **kwargs):
            prs = ArgumentParser(
                prog=prog_name,
                description=dscr,
            )

            prs = get_args(prs)
            args = prs.parse_args()
            function(args)

        return new_function

    return decorator

def ddp_setup():
    init_process_group(backend="nccl")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))

def cleanup_ddp():
    destroy_process_group()

def get_rank():
    return int(os.environ.get("RANK", 0))

def is_rank_0():
    return get_rank() == 0

def print_rank_0(*args, **kwargs):
    if is_rank_0():
        print(*args, **kwargs)