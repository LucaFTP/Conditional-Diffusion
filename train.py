import os
import json
import torch
from argparse import ArgumentParser, Namespace

from trainer import Trainer
from u_net import AttentionUNet
from diffusion_process import DiffusionModel
from gen_utils import (
    parser,
    ddp_setup,
    cleanup_ddp,
    print_rank_0,
)

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
        required=False,
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
    trainer_config = config_file.get("trainer_config")
    diffusion_config = config_file.get("diffusion_config")

    model = AttentionUNet(
        dim=unet_config.get("input"),
        channels=unet_config.get("channels"),
        dim_mults=tuple(unet_config.get("dim_mults")),
    )

    diffusion_model = DiffusionModel(
        model,
        image_size=dataset_config.get("image_size"),
        beta_scheduler=diffusion_config.get("betas_scheduler"),
        timesteps=diffusion_config.get("timesteps"),
        auto_normalize=diffusion_config.get("auto_normalize"),
        verbose=False,
        class_free_par=diffusion_config.get("class_free_par"),
    )

    use_ddp = torch.cuda.device_count() > 1 and "LOCAL_RANK" in os.environ
    if use_ddp:
        ddp_setup()
        local_rank = int(os.environ["LOCAL_RANK"])
    else:
        local_rank = 0  # fallback if in CPU or single-GPU
    
    trainer = Trainer(
        diffusion_model=diffusion_model,
        folder="/leonardo_scratch/fast/uTS25_Fontana/ALL_ROT_npy_version/1024x1024/",
        results_folder=f'./results/{config_file.get("model_name")}',
        dataset_config=dataset_config,
        local_rank=local_rank,
        train_lr=trainer_config.get("train_lr"),
        total_epochs=trainer_config.get("total_epochs"),
        save_and_sample_every=trainer_config.get("save_and_sample_every"),
        num_samples=trainer_config.get("num_samples"),
    )

    print_rank_0(f"Loaded config from {command_line_args.config_filepath}:")
    print_rank_0(json.dumps(config_file, indent=4))
    if milestone := command_line_args.model_milestone:
        trainer.load(milestone)
        print_rank_0(f"Loaded model from milestone {milestone}")
    trainer.train()
    if use_ddp: cleanup_ddp()

if __name__ == "__main__":
    main()