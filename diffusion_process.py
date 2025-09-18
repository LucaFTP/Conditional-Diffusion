from tqdm import tqdm

import torch
import torch.nn as nn
from torch.functional import F

from schedulers import (
    linear_beta_schedule,
    cosine_beta_schedule,
    sigmoid_beta_schedule,
)
from gen_utils import (
    identity,
    normalize_to_neg_one_to_one,
    unnormalize_to_zero_to_one,
    extract,
)

class DiffusionModel(nn.Module):
    SCHEDULER_MAPPING = {
        "linear": linear_beta_schedule,
        "cosine": cosine_beta_schedule,
        "sigmoid": sigmoid_beta_schedule,
    }

    def __init__(
        self,
        model: nn.Module,
        image_size: int,
        *,
        beta_scheduler: str = "cosine",
        timesteps: int = 1000,
        schedule_fn_kwargs: dict | None = None,
        auto_normalize: bool = True,
        verbose: bool = True,
        class_free_par : dict = {}
    ) -> None:
        super().__init__()
        self.model = model

        self.channels = self.model.channels
        self.image_size = image_size

        self.beta_scheduler_fn = self.SCHEDULER_MAPPING.get(beta_scheduler)
        if self.beta_scheduler_fn is None:
            raise ValueError(f"unknown beta schedule {beta_scheduler}")

        if schedule_fn_kwargs is None:
            schedule_fn_kwargs = {}

        betas = self.beta_scheduler_fn(timesteps, **schedule_fn_kwargs)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
        posterior_variance = (
            betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )

        register_buffer = lambda name, val: self.register_buffer(
            name, val.to(torch.float32)
        )

        register_buffer("betas", betas)
        register_buffer("alphas_cumprod", alphas_cumprod)
        register_buffer("alphas_cumprod_prev", alphas_cumprod_prev)
        register_buffer("sqrt_recip_alphas", torch.sqrt(1.0 / alphas))
        register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        register_buffer(
            "sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod)
        )
        register_buffer("posterior_variance", posterior_variance)

        timesteps, *_ = betas.shape
        self.num_timesteps = int(timesteps)

        self.verbose = verbose

        self.normalize = normalize_to_neg_one_to_one if auto_normalize else identity
        self.unnormalize = unnormalize_to_zero_to_one if auto_normalize else identity

        # Classifier free guidance parameters
        self.null_condition  = class_free_par.get("null_condition", -5.0)
        self.p_unconditioned = class_free_par.get("p_unconditioned", 0.1)
        self.guidance_weight = class_free_par.get("guidance_weight", 2.0)

    @torch.inference_mode()
    def p_sample(self, x: torch.Tensor, timestamp: int, mass: torch.Tensor) -> torch.Tensor:
        b, *_, device = *x.shape, x.device
        batched_timestamps = torch.full(
            (2*b,), timestamp, device=device, dtype=torch.long
        )

        x_cat = x.repeat(2, 1, 1, 1)
        mass_cat = torch.cat([mass, torch.full_like(mass, self.null_condition)], dim=0)

        preds_cat = self.model(x_cat, batched_timestamps, mass_cat)
        preds_cond, preds_uncond = preds_cat.chunk(2, dim=0)

        eps = preds_uncond + self.guidance_weight * (preds_cond - preds_uncond)

        t_batch = torch.full((b,), timestamp, device=device, dtype=torch.long)
        betas_t = extract(self.betas, t_batch, x.shape)
        sqrt_recip_alphas_t = extract(
            self.sqrt_recip_alphas, t_batch, x.shape
        )
        sqrt_one_minus_alphas_cumprod_t = extract(
            self.sqrt_one_minus_alphas_cumprod, t_batch, x.shape
        )

        predicted_mean = sqrt_recip_alphas_t * (
            x - betas_t * eps / sqrt_one_minus_alphas_cumprod_t
        )

        if timestamp == 0:
            return predicted_mean
        else:
            posterior_variance = extract(
                self.posterior_variance, t_batch, x.shape
            )
            noise = torch.randn_like(x)
            return predicted_mean + torch.sqrt(posterior_variance) * noise

    @torch.inference_mode()
    def p_sample_loop(
        self, shape: tuple
    ) -> torch.Tensor:
        batch, device = shape[0], "cuda" if torch.cuda.is_available() else "cpu"

        img = torch.randn(shape, device=device)
        mass = 10**(torch.rand((batch, 1, 1, 1), device=device))

        iterator = reversed(range(0, self.num_timesteps))
        if self.verbose: iterator = tqdm(iterator, total=self.num_timesteps)

        with torch.no_grad():
            for t in iterator:
                img = self.p_sample(img, t, mass)

        return self.unnormalize(img), mass
    
    @torch.inference_mode()
    def ddim_sample(
        self, x: torch.Tensor, mass: torch.Tensor, t: int, t_next: int, eta: float = 0.0
    ) -> torch.Tensor:
        b, *_, device = *x.shape, x.device
        batched_timestamps = torch.full(
            (2*b,), t, device=device, dtype=torch.long
        )

        x_cat = x.repeat(2, 1, 1, 1)
        mass_cat = torch.cat([mass, torch.full_like(mass, self.null_condition)], dim=0)

        preds_cat = self.model(x_cat, batched_timestamps, mass_cat)
        preds_cond, preds_uncond = preds_cat.chunk(2, dim=0)

        eps = preds_uncond + self.guidance_weight * (preds_cond - preds_uncond)

        t_batch = torch.full((b,), t, device=device, dtype=torch.long)
        t_next_batch = torch.full((b,), max(int(t_next), 0), device=device, dtype=torch.long)

        alpha_bar_t = extract(self.alphas_cumprod, t_batch, x.shape)
        sqrt_alphas_cumprod = extract(
            self.sqrt_alphas_cumprod, t_batch, x.shape
        )
        sqrt_one_minus_alphas_cumprod_t = extract(
            self.sqrt_one_minus_alphas_cumprod, t_batch, x.shape
        )

        x0_pred = (x - sqrt_one_minus_alphas_cumprod_t * eps) / sqrt_alphas_cumprod

        if t_next < 0:
            alpha_bar_next = torch.ones_like(alpha_bar_t)
        else:
            alpha_bar_next = extract(self.alphas_cumprod, t_next_batch, x.shape)

        if eta == 0 or t_next < 0:
            sigma_t = torch.zeros_like(alpha_bar_t)
            noise = torch.zeros_like(x)
        else:
            sigma_t = eta * torch.sqrt(
                torch.clamp((1 - alpha_bar_t / alpha_bar_next) * (1 - alpha_bar_next) / (1 - alpha_bar_t), min=0.0)
            )
            noise = torch.randn_like(x)

        img = (
            torch.sqrt(alpha_bar_next) * x0_pred
            + torch.sqrt(torch.clamp(1 - alpha_bar_next - sigma_t**2, min=0.0)) * eps
            + sigma_t * noise
        )
        return img

    @torch.inference_mode()
    def ddim_sample_loop(
        self,
        shape: tuple,
        num_steps: int = 50,
        eta: float = 0.0
    ) -> torch.Tensor:
        """
        DDIM sampling
        Args:
            shape: output shape (batch, C, H, W)
            num_steps: number of sampling steps (<< self.num_timesteps)
            eta: amount of stochasticity (0.0 = deterministic, >0 adds noise)
        """
        batch, device = shape[0], "cuda" if torch.cuda.is_available() else "cpu"

        img = torch.randn(shape, device=device)
        mass = 1.5 * 10**(torch.rand((batch, 1, 1, 1), device=device))

        times = torch.linspace(0, self.num_timesteps-1, steps=num_steps).long().tolist()
        times_next = [-1] + times[:-1]

        iterator = zip(reversed(times), reversed(times_next))
        if self.verbose: iterator = tqdm(iterator, total=num_steps)
        
        with torch.no_grad():
            for t, t_next in iterator:
                img = self.ddim_sample(img, mass, t, t_next, eta)

        return self.unnormalize(img), mass

    def sample(
        self, batch_size: int = 16, mode: str = "p"
    ) -> torch.Tensor:
        shape = (batch_size, self.channels, self.image_size, self.image_size)
        if mode == "p":
            return self.p_sample_loop(shape)
        elif mode == "ddim":
            return self.ddim_sample_loop(shape)
        else:
            raise ValueError(f"Unknown mode: {mode}")

    def q_sample(
        self, x_start: torch.Tensor, t: int, noise: torch.Tensor = None
    ) -> torch.Tensor:
        if noise is None:
            noise = torch.randn_like(x_start)

        sqrt_alphas_cumprod_t = extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alphas_cumprod_t = extract(
            self.sqrt_one_minus_alphas_cumprod, t, x_start.shape
        )

        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise
    
    def p_loss(
        self,
        x_start: torch.Tensor,
        t: int,
        mass: torch.Tensor,
        noise: torch.Tensor = None,
        loss_type: str = "l2",
    ) -> torch.Tensor:
        if noise is None:
            noise = torch.randn_like(x_start)
        x_noised = self.q_sample(x_start, t, noise=noise)
        predicted_noise = self.model(x_noised, t, mass)

        if loss_type == "l2":
            loss = F.mse_loss(noise, predicted_noise)
        elif loss_type == "l1":
            loss = F.l1_loss(noise, predicted_noise)
        else:
            raise ValueError(f"unknown loss type {loss_type}")
        return loss

    def forward(self, x: torch.Tensor, mass: torch.Tensor) -> torch.Tensor:
        b, c, h, w, device, img_size = *x.shape, x.device, self.image_size
        assert h == w == img_size, f"image size must be {img_size}"

        timestamps = torch.randint(0, self.num_timesteps, (b,), device=device).long()
        x = self.normalize(x)

        if torch.rand((1,)) < self.p_unconditioned:
            mass = torch.full_like(mass, self.null_condition)

        return self.p_loss(x, timestamps, mass)