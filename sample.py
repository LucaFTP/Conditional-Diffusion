import os
import torch
import numpy as np
from argparse import ArgumentParser, Namespace

from sz_diffusion.diffusion_process import DiffusionModel
from sz_diffusion.gen_utils import (
    parser,
    save_img_grid,
    tensor_to_numpy,
    ddp_setup,
    cleanup_ddp
)

import torch.distributed as dist

def parse_arguments(parser: ArgumentParser) -> ArgumentParser:
    parser.add_argument(
        "--model-name",
        help="The name of the model to load",
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
        "-o",
        "--output-type",
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
    parser.add_argument(
        "-t",
        "--type-of-sampling",
        help="The type of sampling to use (default: 'p', options: 'p' <probabilistic>, 'ddim' <implicit>)",
        default="p",
        choices=["p", "ddim"],
    )
    return parser

@parser(
    "Diffusion model training script",
    "Diffusion model training script",
    parse_arguments,
)
def main(command_line_args: Namespace) -> None:

    # torch.manual_seed(2312)
    model_name = command_line_args.model_name

    use_ddp = torch.cuda.device_count() > 1 and "LOCAL_RANK" in os.environ
    if use_ddp:
        ddp_setup()
        local_rank = int(os.environ["LOCAL_RANK"])
    else:
        local_rank = 0  # fallback if in CPU or single-GPU
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    diffusion_model = DiffusionModel.from_pretrained(
        checkpoint_path=str(f"/leonardo_work/uTS25_Fontana/Diffusion_ckpt/{model_name}/model-{command_line_args.model_milestone}.pt"),
        device=device,
        use_ema=True,
    )

    if command_line_args.output_type == "png":
        if local_rank == 0:
            samples, sample_masses = diffusion_model.sample(batch_size=command_line_args.num_samples, mode=command_line_args.type_of_sampling)
            save_img_grid(tensor_to_numpy(samples), tensor_to_numpy(sample_masses), 
            model_name=model_name, milestone=command_line_args.model_milestone, cbar=True)

    if command_line_args.output_type == "npy":
        from tqdm import tqdm
        world_size = dist.get_world_size() if dist.is_initialized() else 1

        total_samples = command_line_args.num_samples
        samples_per_rank = total_samples // world_size
        if local_rank < total_samples % world_size:
            samples_per_rank += 1

        batch_size = 64
        steps = (samples_per_rank + batch_size - 1) // batch_size

        out_dir = f"results/{model_name}/"
        if local_rank == 0:
            os.makedirs(out_dir, exist_ok=True)

        if use_ddp: dist.barrier()  # Ensure directory is created before other ranks proceed

        for i in tqdm(range(steps), desc=f"Rank {local_rank}", position=local_rank, leave=False):
            
            current_bs = min(batch_size, samples_per_rank - i * batch_size)

            samples, sample_masses = diffusion_model.sample(
                batch_size=current_bs,
                mode=command_line_args.type_of_sampling
            )

            np.save(f"{out_dir}/samples-{command_line_args.model_milestone}-rank{local_rank}-{i}.npy", tensor_to_numpy(samples))
            np.save(f"{out_dir}/masses-{command_line_args.model_milestone}-rank{local_rank}-{i}.npy", tensor_to_numpy(sample_masses))

    if use_ddp: cleanup_ddp()

if __name__ == "__main__":
    main()