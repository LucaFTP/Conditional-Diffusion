import os
import json
import torch
import numpy as np
from argparse import ArgumentParser, Namespace

from u_net import AttentionUNet
from diffusion_process import DiffusionModel
from gen_utils import (
    parser,
    save_img_grid,
    tensor_to_numpy,
    ddp_setup,
    cleanup_ddp
)

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

def parse_arguments(parser: ArgumentParser) -> ArgumentParser:
    parser.add_argument(
        "-c",
        "--config-filepath",
        help="The config filepath for the model/trainer config (Litteral filepath form this file)",
        type=str,
        required=True,
    )
    parser.add_argument(
        "-m",
        "--model-milestone",
        help="The milestone of the model",
        type=int,
        required=True,
    )
    parser.add_argument(
        "-t",
        "--type-of-output",
        help="The type of output to save (default: 'png', options: 'npy', 'png'). According to the type, choose wisely the number of samples to generate.",
        default="png",
        choices=["npy", "png"],
    )
    parser.add_argument(
        "-n",
        "--num-samples",
        help="The number of image to sample from the model",
        type=int,
        required=True,
    )
    return parser

@parser(
    "Diffusion model training script",
    "Diffusion model training script",
    parse_arguments,
)
def main(command_line_args: Namespace) -> None:

    if not os.path.isfile(command_line_args.config_filepath):
        raise FileNotFoundError(command_line_args.config_filepath)

    with open(command_line_args.config_filepath, "r") as f:
        config_file = json.load(f)

    unet_config = config_file.get("unet_config")
    dataset_config = config_file.get("dataset_config")
    diffusion_config = config_file.get("diffusion_config")

    use_ddp = torch.cuda.device_count() > 1 and "LOCAL_RANK" in os.environ
    if use_ddp:
        ddp_setup()
        local_rank = int(os.environ["LOCAL_RANK"])
    else:
        local_rank = 0  # fallback if in CPU or single-GPU
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load(
        str(f"./results/{config_file.get('model_name')}/model-{command_line_args.model_milestone}.pt"),
        map_location=device,
    )
    model = AttentionUNet(
        dim=unet_config.get("input"),
        channels=unet_config.get("channels"),
        dim_mults=tuple(unet_config.get("dim_mults")),
    ).to(device)
    model.load_state_dict(checkpoint["model"])
    if dist.is_initialized():
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    diffusion_model = DiffusionModel(
        model.module if isinstance(model, DDP) else model,
        image_size=dataset_config.get("image_size"),
        beta_scheduler=diffusion_config.get("betas_scheduler"),
        timesteps=diffusion_config.get("timesteps"),
        auto_normalize=diffusion_config.get("auto_normalize"),
    ).to(device)

    if command_line_args.type_of_output == "png":
        samples, sample_masses = diffusion_model.sample(batch_size=command_line_args.num_samples)
        save_img_grid(tensor_to_numpy(samples), tensor_to_numpy(sample_masses), model_name=config_file.get("model_name"), milestone=command_line_args.model_milestone, cbar=True)
    if command_line_args.type_of_output == "npy":
        from tqdm import tqdm
        world_size = dist.get_world_size() if dist.is_initialized() else 1

        total_samples = command_line_args.num_samples
        samples_per_rank = total_samples // world_size
        if local_rank < total_samples % world_size:
            samples_per_rank += 1

        batch_size = 64
        steps = samples_per_rank // batch_size + int(samples_per_rank % batch_size > 0)

        out_dir = f"results/{config_file.get('model_name')}/"
        os.makedirs(out_dir, exist_ok=True)

        for i in tqdm(range(steps), desc=f"Rank {local_rank} sampling"):
            current_bs = batch_size if i < steps - 1 else samples_per_rank % batch_size or batch_size
            samples, sample_masses = diffusion_model.sample(batch_size=current_bs)

            np.save(f"{out_dir}/samples-{command_line_args.model_milestone}-rank{local_rank}-{i}.npy", tensor_to_numpy(samples))
            np.save(f"{out_dir}/masses-{command_line_args.model_milestone}-rank{local_rank}-{i}.npy", tensor_to_numpy(sample_masses))

    if use_ddp: cleanup_ddp()

if __name__ == "__main__":
    main()