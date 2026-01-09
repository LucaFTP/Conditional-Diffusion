from tqdm import tqdm

import torch
import torch.nn as nn
from torch.functional import F

from sz_diffusion.schedulers import (
    linear_beta_schedule,
    cosine_beta_schedule,
    sigmoid_beta_schedule,
)
from sz_diffusion.gen_utils import (
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
        
    @classmethod
    def from_pretrained(cls, checkpoint_path, device='cpu', use_ema=True):

        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        unet_config = checkpoint.get('unet_config')
        dataset_config = checkpoint.get('dataset_config')
        diffusion_config = checkpoint.get('diffusion_config')
        
        if unet_config is None or diffusion_config is None:
            raise ValueError("Checkpoint missing configurations.")
        
        from ema_pytorch import EMA
        from sz_diffusion.u_net import AttentionUNet

        unet = AttentionUNet(
            dim=unet_config.get("input"),
            channels=unet_config.get("channels"),
            dim_mults=tuple(unet_config.get("dim_mults")),
        )
        
        diffusion_instance = cls(
            model=unet,
            image_size=dataset_config.get("image_size"),
            beta_scheduler=diffusion_config.get("betas_scheduler"),
            timesteps=diffusion_config.get("timesteps"),
            auto_normalize=diffusion_config.get("auto_normalize"),
        ).to(device)

        if use_ema and 'ema' in checkpoint:
            print("Loading EMA weights into the model...")
            # Using a temporary EMA instance to load the weights
            # in order to use the DiffusionModel class structure
            # for inference.
            ema_temp = EMA(diffusion_instance, beta=0.995)
            ema_temp.load_state_dict(checkpoint['ema'])
            diffusion_instance.model.load_state_dict(ema_temp.ema_model.model.state_dict())
            
            del ema_temp
        else:
            print("Loading standard training weights...")
            diffusion_instance.model.load_state_dict(checkpoint['model'])\
        
        diffusion_instance.dataset_config = dataset_config
        
        return diffusion_instance

    @torch.inference_mode()
    def p_sample_loop(
        self, shape: tuple
    ) -> torch.Tensor:
        batch, device = shape[0], "cuda" if torch.cuda.is_available() else "cpu"

        img = torch.randn(shape, device=device)
        mass = 10 ** ( torch.rand(
            (batch, 1, 1, 1), dtype=torch.float32, device=device
            ) * (15.2 - 13.8) )

        iterator = reversed(range(0, self.num_timesteps))
        if self.verbose: iterator = tqdm(iterator, total=self.num_timesteps)
        
        for t in iterator:
            img = self.p_sample(img, t, mass)

        return self.unnormalize(img), mass
    
    @torch.inference_mode()
    def dpm_solver_step(self, x, t, s, mass, order=2):
        """
        Executes one step of DPM-Solver (1° o 2° ordine).
        x: current image
        t: current timestep (index in self.alphas_cumprod)
        s: next timestep (smaller than t)
        mass: conditioning
        order: order of the solver (1 or 2)
        """
        b, device = x.shape[0], x.device
        t_batch = torch.full((b,), t, device=device, dtype=torch.long)
        s_batch = torch.full((b,), s, device=device, dtype=torch.long)

        # log alphas
        at  = extract(self.alphas_cumprod, t_batch, x.shape)  # α_t_bar
        as_ = extract(self.alphas_cumprod, s_batch, x.shape)  # α_s_bar
        log_at = torch.log(at)
        log_as = torch.log(as_)

        x_in = x.repeat(2,1,1,1)
        mass_in = torch.cat([mass, torch.full_like(mass, self.null_condition)], dim=0)
        t_in = torch.full((2*b,), t, device=device, dtype=torch.long)

        preds = self.model(x_in, t_in, mass_in)
        eps_cond, eps_uncond = preds.chunk(2, dim=0)
        eps = eps_uncond + self.guidance_weight * (eps_cond - eps_uncond)

        # x0 and derivative prediction
        x0_pred = (x - torch.sqrt(1 - at) * eps) / torch.sqrt(at)

        if order == 1 or s == 0:
            # DPM-Solver-1
            x_next = torch.sqrt(as_) * x0_pred + torch.sqrt(1 - as_) * eps
            return x_next

        elif order == 2:
            # DPM-Solver-2 (second order)
            # intermediate prediction at half logSNR
            log_h = log_as - log_at
            log_mid = log_at + 0.5 * log_h
            a_mid = torch.exp(log_mid)

            x_mid = torch.sqrt(a_mid) * x0_pred + torch.sqrt(1 - a_mid) * eps

            # prediction of eps at mid-point
            x_in = x_mid.repeat(2,1,1,1)
            mass_in = torch.cat([mass, torch.full_like(mass, self.null_condition)], dim=0)
            t_in = torch.full((2*b,), (t+s)//2, device=device, dtype=torch.long)

            preds_mid = self.model(x_in, t_in, mass_in)
            eps_cond_mid, eps_uncond_mid = preds_mid.chunk(2, dim=0)
            eps_mid = eps_uncond_mid + self.guidance_weight * (eps_cond_mid - eps_uncond_mid)

            # final estimate
            x0_pred_mid = (x_mid - torch.sqrt(1 - a_mid) * eps_mid) / torch.sqrt(a_mid)
            x_next = torch.sqrt(as_) * x0_pred_mid + torch.sqrt(1 - as_) * eps_mid
            return x_next

        else:
            raise ValueError("order must be 1 or 2")
        
    @torch.inference_mode()
    def sample_dpm_solver(self, shape, steps=20, order=2):
        """
        Sampling with DPM-Solver.
        shape: (batch, channels, H, W)
        steps: number of steps (e.g. 20 or 50)
        order: order of the solver (1 or 2)
        """
        batch, device = shape[0], "cuda" if torch.cuda.is_available() else "cpu"
        img = torch.randn(shape, device=device)
        mass = 10 ** ( torch.rand(
            (batch, 1, 1, 1), dtype=torch.float32, device=device
            ) * (15.2 - 13.8) + 13.8 )

        seq = torch.linspace(0, self.num_timesteps-1, steps, dtype=int)
        seq = list(reversed(seq))

        for i in tqdm(range(len(seq)-1)):
            t, s = seq[i], seq[i+1]
            img = self.dpm_solver_step(img, t, s, mass, order=order)

        return self.unnormalize(img), mass

    def sample(
        self, batch_size: int = 16, mode: str = "p"
    ) -> torch.Tensor:
        shape = (batch_size, self.channels, self.image_size, self.image_size)
        if mode == "p":
            return self.p_sample_loop(shape)
        elif mode == "ddim":
            return self.sample_dpm_solver(shape, steps=100, order=2)
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