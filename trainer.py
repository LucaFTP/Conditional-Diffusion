from tqdm import tqdm
from pathlib import Path

import torch
from ema_pytorch import EMA
from torchvision import utils
from torch.optim import AdamW
from torch.utils.tensorboard import SummaryWriter

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from gen_utils import cycle
from dataset import create_dataloader
from diffusion_process import DiffusionModel

class Trainer:
    def __init__(
        self,
        diffusion_model: DiffusionModel,
        folder: str,
        results_folder: str,
        dataset_config: dict,
        local_rank: int,
        *,
        train_lr: float = 1e-4,
        train_num_steps: int = 10000,
        ema_update_every: int = 10,
        ema_decay: float = 0.995,
        adam_betas: tuple[float, float] = (0.9, 0.99),
        save_and_sample_every: int = 1000,
        num_samples: int = 4,
        save_best_and_latest_only: bool = False,
    ) -> None:
        self.local_rank = local_rank
        self.device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

        self.d_model = diffusion_model.to(self.device)
        if dist.is_initialized():
            self.d_model.model = DDP(self.d_model.model, device_ids=[local_rank], output_device=local_rank)

        self.channels = diffusion_model.channels
        self.step = 0
        self.num_samples = num_samples
        self.save_and_sample_every = save_and_sample_every
        self.train_num_steps = train_num_steps
        self.image_size = diffusion_model.image_size

        self.ds, self.dl_sampler, self.dl = create_dataloader(
            root_dir=folder,
            dataset_config=dataset_config
        )
        self.dl = cycle(self.dl)

        self.opt = AdamW(diffusion_model.parameters(), lr=train_lr, betas=adam_betas)
        self.ema = EMA(diffusion_model, beta=ema_decay, update_every=ema_update_every)
        self.ema.to(self.device)

        self.results_folder = Path(results_folder)
        self.results_folder.mkdir(exist_ok=True)
        self.image_folder = self.results_folder / "ema_sampling"
        self.image_folder.mkdir(exist_ok=True)
        self.save_best_and_latest_only = save_best_and_latest_only
        self.writer = SummaryWriter(str(self.results_folder)) if dist.get_rank() == 0 else None

    def save(self, milestone: int) -> None:
        if dist.get_rank() != 0:
            return

        data = {
            "step": self.step,
            "model": self.d_model.model.module.state_dict(),
            "opt": self.opt.state_dict(),
            "ema": self.ema.state_dict(),
            "version": "1.0",
        }
        torch.save(data, str(self.results_folder / f"model-{milestone}.pt"))

    def load(self, milestone: int) -> None:
        map_location = {f"cuda:{i}": f"cuda:{self.local_rank}" for i in range(torch.cuda.device_count())}
        data = torch.load(str(self.results_folder / f"model-{milestone}.pt"), map_location=map_location)

        self.d_model.model.module.load_state_dict(data["model"])
        self.step = data["step"]
        self.opt.load_state_dict(data["opt"])
        self.ema.load_state_dict(data["ema"])

        if "version" in data:
            print(f"loading from version {data['version']}")

    def train(self) -> None:
        try:
            with tqdm(initial=self.step, total=self.train_num_steps, disable=dist.get_rank() != 0) as pbar:
                while self.step < self.train_num_steps:
                    self.dl_sampler.set_epoch(self.step)

                    total_loss = 0.0
                    data = next(self.dl)
                    image = data[0].clone().detach().to(self.device, non_blocking=True)
                    mass  = data[1].clone().detach().to(self.device, non_blocking=True)
                    
                    loss = self.d_model(image, mass)
                    total_loss += loss.item()

                    loss.backward()
                    self.opt.step()
                    self.opt.zero_grad()

                    self.step += 1
                    self.ema.update()

                    if dist.get_rank() == 0:
                        pbar.set_description(f"loss: {total_loss:.4f}")
                        self.writer.add_scalar("Loss/train", total_loss, self.step)

                        if self.step % self.save_and_sample_every == 0:
                            self.ema.ema_model.eval()
                            with torch.inference_mode():
                                milestone = self.step // self.save_and_sample_every
                                sampled_imgs, _ = self.ema.ema_model.sample(batch_size=self.num_samples)

                            utils.save_image(
                                sampled_imgs,
                                str(self.image_folder / f"sample-{milestone}.png"),
                                nrow=int(self.num_samples**0.5)
                            )
                            self.save(milestone)
                    if dist.get_rank() == 0:
                        pbar.update(1)
        finally:
            if self.writer:
                self.writer.close()