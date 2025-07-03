from tqdm import tqdm
from pathlib import Path

import torch
from ema_pytorch import EMA
from torchvision import utils
from torch.optim import AdamW
from torch.utils.tensorboard import SummaryWriter

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
        self.d_model = diffusion_model
        self.channels = diffusion_model.channels

        self.step = 0

        self.num_samples = num_samples
        self.save_and_sample_every = save_and_sample_every

        self.train_num_steps = train_num_steps
        self.image_size = diffusion_model.image_size

        train_loader = create_dataloader(
            root_dir=folder,
            dataset_config=dataset_config
        )
        self.dl = cycle(train_loader)

        self.opt = AdamW(diffusion_model.parameters(), lr=train_lr, betas=adam_betas)

        self.ema = EMA(diffusion_model, beta=ema_decay, update_every=ema_update_every)
        self.ema.to(self.device)

        self.results_folder = Path(results_folder)
        self.results_folder.mkdir(exist_ok=True)
        self.save_best_and_latest_only = save_best_and_latest_only
        self.writer = SummaryWriter(results_folder)

    @property
    def device(self) -> str:
        return "cuda" if torch.cuda.is_available() else "cpu"

    def save(self, milestone: int) -> None:
        data = {
            "step": self.step,
            "model": self.d_model.model.state_dict(),
            "opt": self.opt.state_dict(),
            "ema": self.ema.state_dict(),
            "version": "1.0",
        }

        torch.save(data, str(self.results_folder / f"model-{milestone}.pt"))

    def load(self, milestone: int) -> None:
        data = torch.load(
            str(self.results_folder / f"model-{milestone}.pt"),
            map_location=self.device,
        )
        self.d_model.model.load_state_dict(data["model"])

        self.step = data["step"]
        self.opt.load_state_dict(data["opt"])
        self.ema.load_state_dict(data["ema"])

        if "version" in data:
            print(f"loading from version {data['version']}")

    def ema_sample(
            self, num_samples: int = 16
    ) -> torch.Tensor:
        
        sampled_imgs = self.ema.ema_model.sample(
            batch_size = num_samples
        )
        return sampled_imgs

    def train(self) -> None:
        try:
            with tqdm(initial=self.step, total=self.train_num_steps) as pbar:
                while self.step < self.train_num_steps:
                    total_loss = 0.0

                    data = next(self.dl)
                    image = data[0].clone().detach().to(self.device, non_blocking=True)
                    mass  = data[1].clone().detach().to(self.device, non_blocking=True)

                    loss = self.d_model(image, mass)
                    total_loss += loss.item()

                    loss.backward()

                    pbar.set_description(f"loss: {total_loss:.4f}")
                    self.writer.add_scalar("Loss/train", total_loss, self.step)

                    self.opt.step()
                    self.opt.zero_grad()

                    self.step += 1
                    self.ema.update()

                    if self.step != 0 and self.step % self.save_and_sample_every == 0:
                        self.ema.ema_model.eval()

                        with torch.inference_mode():
                            milestone = self.step // self.save_and_sample_every
                            sampled_imgs, _ = self.ema.ema_model.sample(
                                batch_size=self.num_samples
                            )

                        utils.save_image(
                            sampled_imgs,
                            str(self.results_folder / f"ema_sampling/sample-{milestone}.png"),
                            nrow=int(self.num_samples**0.5)
                        )

                        self.save(milestone)
                    pbar.update(1)
        finally:
            self.writer.close()