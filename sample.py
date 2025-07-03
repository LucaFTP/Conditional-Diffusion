import os
import json
import time
import numpy as np
from argparse import ArgumentParser, Namespace

import torch
from torchvision import utils

from u_net import AttentionUNet
from diffusion_process import DiffusionModel
from gen_utils import parser, save_img_grid, tensor_to_numpy

def parse_arguments(parser: ArgumentParser) -> ArgumentParser:
    parser.add_argument(
        "-c",
        "--config-filepath",
        help="The config filepath for the model/trainer config (Litteral filepath form this file)",
        type=str,
        required=True,
    )
    parser.add_argument(
        "-n",
        "--num-samples",
        help="The number of image to sample from the model",
        type=int,
        required=True,
    )
    parser.add_argument(
        "-m",
        "--model-milestone",
        help="The milestone of the model",
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

    device = "cuda" if torch.cuda.is_available() else "cpu"

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

    diffusion_model = DiffusionModel(
        model,
        image_size=dataset_config.get("image_size"),
        beta_scheduler=diffusion_config.get("betas_scheduler"),
        timesteps=diffusion_config.get("timesteps"),
        auto_normalize=diffusion_config.get("auto_normalize"),
    ).to(device)

    # samples, sample_masses = diffusion_model.sample(batch_size=command_line_args.num_samples)
    # save_img_grid(tensor_to_numpy(samples), tensor_to_numpy(sample_masses), model_name=config_file.get("model_name"), milestone=command_line_args.model_milestone, cbar=True)
    from tqdm import tqdm
    for i in tqdm(range(command_line_args.num_samples // 64), desc="Sampling images"):
        samples, sample_masses = diffusion_model.sample(batch_size=64)
        np.save(f"results/{config_file.get('model_name')}/samples-{command_line_args.model_milestone}-{i}.npy", tensor_to_numpy(samples))
        np.save(f"results/{config_file.get('model_name')}/masses-{command_line_args.model_milestone}-{i}.npy", tensor_to_numpy(sample_masses))

    '''
    for ix, sample in enumerate(samples):
        utils.save_image(sample, f"sample-{time.strftime('%Y-%m-%d-%s')}-{ix}.png")
    '''

if __name__ == "__main__":
    main()