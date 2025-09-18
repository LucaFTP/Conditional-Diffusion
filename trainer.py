from tqdm import tqdm
from pathlib import Path

import torch
from ema_pytorch import EMA
from torchvision import utils
from torch.optim import AdamW
from torch.utils.tensorboard import SummaryWriter

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from gen_utils import print_rank_0
from dataset import create_dataloader
from diffusion_process import DiffusionModel

class Trainer:
    def __init__(
        self,
        diffusion_model: DiffusionModel,
        folder: str,
        model_name: str,
        dataset_config: dict,
        local_rank: int,
        *,
        train_lr: float = 1e-4,
        total_epochs: int = 100,
        ema_update_every: int = 10,
        ema_decay: float = 0.995,
        adam_betas: tuple[float, float] = (0.9, 0.99),
        save_and_sample_every: int = 10,
        num_samples: int = 4,
        save_best_and_latest_only: bool = False,
    ) -> None:
        self.local_rank = local_rank
        self.device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

        self.d_model = diffusion_model.to(self.device)
        self.ema = EMA(diffusion_model, beta=ema_decay, update_every=ema_update_every)
        self.ema.to(self.device)

        if dist.is_initialized():
            self.d_model = DDP(self.d_model, device_ids=[local_rank], output_device=local_rank)
        
        self.opt = AdamW(self.d_model.parameters(), lr=train_lr, betas=adam_betas)

        self.ds, self.dl_sampler, self.dl = create_dataloader(
            root_dir=folder,
            dataset_config=dataset_config
        )
        
        self.start_epoch = 1
        self.total_epochs = total_epochs

        self.results_folder = Path(f'./results/{model_name}')
        self.results_folder.mkdir(exist_ok=True)
        self.image_folder = self.results_folder / "ema_sampling"
        self.image_folder.mkdir(exist_ok=True)
        self.checkpoint_folder = Path(f"/leonardo_work/uTS25_Fontana/Diffusion_ckpt/{model_name}")
        self.checkpoint_folder.mkdir(exist_ok=True)
        self.num_samples = num_samples
        self.save_and_sample_every = save_and_sample_every
        self.save_best_and_latest_only = save_best_and_latest_only
        self.writer = SummaryWriter(str(self.results_folder)) if dist.get_rank() == 0 else None

    def save(self, milestone: int) -> None:
        if dist.get_rank() != 0:
            return

        data = {
            "epoch": milestone * self.save_and_sample_every,
            "model": self.d_model.module.model.state_dict(),
            "opt": self.opt.state_dict(),
            "ema": self.ema.state_dict(),
            "version": "1.0",
        }
        torch.save(data, str(self.checkpoint_folder / f"model-{milestone}.pt"))

    def load(self, milestone: int) -> None:
        map_location = {f"cuda:{i}": f"cuda:{self.local_rank}" for i in range(torch.cuda.device_count())}
        data = torch.load(str(self.checkpoint_folder / f"model-{milestone}.pt"), map_location=map_location)

        self.d_model.module.model.load_state_dict(data["model"])
        self.start_epoch = data["epoch"]
        self.opt.load_state_dict(data["opt"])
        self.ema.load_state_dict(data["ema"])

        if "version" in data:
            print_rank_0(f"loading from version {data['version']}")

    def train(self) -> None:
        try:
            pbar = tqdm(initial=self.start_epoch, total=self.total_epochs, disable=dist.get_rank() != 0)
            for epoch in range(self.start_epoch, self.total_epochs+1):
                self.dl_sampler.set_epoch(epoch)
                epoch_loss = 0.0

                for data in self.dl:

                    image = data[0].clone().detach().to(self.device, non_blocking=True)
                    mass  = data[1].clone().detach().to(self.device, non_blocking=True)

                    loss = self.d_model(image, mass)
                    epoch_loss += loss.item()

                    loss.backward()
                    self.opt.step()
                    self.opt.zero_grad(set_to_none=True)

                    self.ema.update()

                if dist.get_rank() == 0:
                    self.writer.add_scalar("Loss/train", epoch_loss / len(self.dl), epoch)

                    if epoch % self.save_and_sample_every == 0:
                        self.ema.ema_model.eval()
                        with torch.inference_mode():
                            milestone = epoch // self.save_and_sample_every
                            sampled_imgs, _ = self.ema.ema_model.sample(batch_size=self.num_samples)

                        utils.save_image(
                            sampled_imgs,
                            str(self.image_folder / f"sample-{milestone}.png"),
                            nrow=int(self.num_samples**0.5)
                        )
                        self.save(milestone)
                        self.writer.add_images("Generated Images", sampled_imgs, epoch)

                pbar.set_description(f"loss: {epoch_loss / len(self.dl):.4f}")
                pbar.update(1)

        finally:
            if self.writer:
                self.writer.close()