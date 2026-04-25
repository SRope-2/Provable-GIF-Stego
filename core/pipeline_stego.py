import numpy as np
import torch
import copy
from PIL import Image
from scipy.stats import norm,truncnorm
from functools import reduce
from scipy.special import betainc
import inspect
from typing import Callable, List, Optional, Union
from dataclasses import dataclass

# [!] 严丝合缝的包导入
from .watermark_core import Gaussian_Shading, Gaussian_Shading_chacha

from tqdm import tqdm
from diffusers.utils import is_accelerate_available
from packaging import version
from transformers import CLIPTextModel, CLIPTokenizer

from diffusers.configuration_utils import FrozenDict
from diffusers.models import AutoencoderKL
from diffusers import DiffusionPipeline
from diffusers.schedulers import (
    DDIMScheduler,
    DPMSolverMultistepScheduler,
    EulerAncestralDiscreteScheduler,
    EulerDiscreteScheduler,
    LMSDiscreteScheduler,
    PNDMScheduler,
)
from diffusers.utils import logging, BaseOutput
from einops import rearrange
from functools import partial

# AnimateDiff 特定模型：运行时按需导入，避免在未安装 AnimateDiff 时报错
# 使用时需将 AnimateDiff 仓库根目录加入 sys.path，或通过 pip install 安装
try:
    from animatediff.models.unet import UNet3DConditionModel
    from animatediff.models.sparse_controlnet import SparseControlNetModel
except ImportError:
    # 如果直接把 AnimateDiff 代码放在项目内（models/ 目录），则尝试该路径
    try:
        from models.unet import UNet3DConditionModel
        from models.sparse_controlnet import SparseControlNetModel
    except ImportError:
        import warnings
        warnings.warn(
            "AnimateDiff models not found. AnimationPipeline requires AnimateDiff. "
            "Please install AnimateDiff: https://github.com/guoyww/AnimateDiff\n"
            "Or add AnimateDiff repo root to sys.path before importing this module.",
            ImportWarning,
            stacklevel=2,
        )
        UNet3DConditionModel = None
        SparseControlNetModel = None

device = 'cuda' if torch.cuda.is_available() else 'cpu'
logger = logging.get_logger(__name__)

def backward_ddim(x_t, alpha_t, alpha_tm1, eps_xt):
    return (
        alpha_tm1**0.5
        * (
            (alpha_t**-0.5 - alpha_tm1**-0.5) * x_t
            + ((1 / alpha_tm1 - 1) ** 0.5 - (1 / alpha_t - 1) ** 0.5) * eps_xt
        )
        + x_t
    )

def forward_ddim(x_t, alpha_t, alpha_tp1, eps_xt):
    return backward_ddim(x_t, alpha_t, alpha_tp1, eps_xt)

@dataclass
class AnimationPipelineOutput(BaseOutput):
    videos: Union[torch.Tensor, np.ndarray]
    init_latents: Optional[torch.FloatTensor]

class AnimationPipeline(DiffusionPipeline):
    _optional_components = []

    def __init__(
        self, vae: AutoencoderKL, text_encoder: CLIPTextModel, tokenizer: CLIPTokenizer,
        unet: UNet3DConditionModel, scheduler: Union[DDIMScheduler, PNDMScheduler, LMSDiscreteScheduler, EulerDiscreteScheduler, EulerAncestralDiscreteScheduler, DPMSolverMultistepScheduler],
        controlnet: Union[SparseControlNetModel, None] = None,
    ):
        super().__init__()
        self.forward_diffusion = partial(self.backward_diffusion, reverse_process=True)
        self.count = 0
        if hasattr(scheduler.config, "steps_offset") and scheduler.config.steps_offset != 1:
            new_config = dict(scheduler.config)
            new_config["steps_offset"] = 1
            scheduler._internal_dict = FrozenDict(new_config)

        if hasattr(scheduler.config, "clip_sample") and scheduler.config.clip_sample is True:
            new_config = dict(scheduler.config)
            new_config["clip_sample"] = False
            scheduler._internal_dict = FrozenDict(new_config)

        is_unet_version_less_0_9_0 = hasattr(unet.config, "_diffusers_version") and version.parse(version.parse(unet.config._diffusers_version).base_version) < version.parse("0.9.0.dev0")
        is_unet_sample_size_less_64 = hasattr(unet.config, "sample_size") and unet.config.sample_size < 64
        if is_unet_version_less_0_9_0 and is_unet_sample_size_less_64:
            new_config = dict(unet.config)
            new_config["sample_size"] = 64
            unet._internal_dict = FrozenDict(new_config)

        self.register_modules(vae=vae, text_encoder=text_encoder, tokenizer=tokenizer, unet=unet, scheduler=scheduler, controlnet=controlnet)
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)

    def enable_vae_slicing(self): self.vae.enable_slicing()
    def disable_vae_slicing(self): self.vae.disable_slicing()

    def enable_sequential_cpu_offload(self, gpu_id=0):
        if is_accelerate_available(): from accelerate import cpu_offload
        else: raise ImportError("Please install accelerate via `pip install accelerate`")
        device = torch.device(f"cuda:{gpu_id}")
        for cpu_offloaded_model in [self.unet, self.text_encoder, self.vae]:
            if cpu_offloaded_model is not None: cpu_offload(cpu_offloaded_model, device)

    @property
    def _execution_device(self):
        if self.device != torch.device("meta") or not hasattr(self.unet, "_hf_hook"): return self.device
        for module in self.unet.modules():
            if hasattr(module, "_hf_hook") and hasattr(module._hf_hook, "execution_device") and module._hf_hook.execution_device is not None:
                return torch.device(module._hf_hook.execution_device)
        return self.device

    def _encode_prompt(self, prompt, device, num_videos_per_prompt, do_classifier_free_guidance, negative_prompt):
        batch_size = len(prompt) if isinstance(prompt, list) else 1
        text_inputs = self.tokenizer(prompt, padding="max_length", max_length=self.tokenizer.model_max_length, truncation=True, return_tensors="pt")
        text_input_ids = text_inputs.input_ids
        untruncated_ids = self.tokenizer(prompt, padding="longest", return_tensors="pt").input_ids

        if hasattr(self.text_encoder.config, "use_attention_mask") and self.text_encoder.config.use_attention_mask:
            attention_mask = text_inputs.attention_mask.to(device)
        else:
            attention_mask = None

        text_embeddings = self.text_encoder(text_input_ids.to(device), attention_mask=attention_mask)[0]
        bs_embed, seq_len, _ = text_embeddings.shape
        text_embeddings = text_embeddings.repeat(1, num_videos_per_prompt, 1).view(bs_embed * num_videos_per_prompt, seq_len, -1)

        if do_classifier_free_guidance:
            uncond_tokens = [""] * batch_size if negative_prompt is None else (negative_prompt if isinstance(negative_prompt, list) else [negative_prompt] * batch_size)
            max_length = text_input_ids.shape[-1]
            uncond_input = self.tokenizer(uncond_tokens, padding="max_length", max_length=max_length, truncation=True, return_tensors="pt")
            attention_mask = uncond_input.attention_mask.to(device) if hasattr(self.text_encoder.config, "use_attention_mask") and self.text_encoder.config.use_attention_mask else None
            uncond_embeddings = self.text_encoder(uncond_input.input_ids.to(device), attention_mask=attention_mask)[0]
            seq_len = uncond_embeddings.shape[1]
            uncond_embeddings = uncond_embeddings.repeat(1, num_videos_per_prompt, 1).view(batch_size * num_videos_per_prompt, seq_len, -1)
            text_embeddings = torch.cat([uncond_embeddings, text_embeddings])

        return text_embeddings

    def decode_latents(self, latents):
        video_length = latents.shape[2]
        latents = 1 / 0.18215 * latents
        latents = rearrange(latents, "b c f h w -> (b f) c h w")
        video = []
        for frame_idx in tqdm(range(latents.shape[0])):
            video.append(self.vae.decode(latents[frame_idx:frame_idx+1]).sample)
        video = torch.cat(video)
        video = rearrange(video, "(b f) c h w -> b c f h w", f=video_length)
        print("video-shape = ",video.shape)
        video = (video / 2 + 0.5).clamp(0, 1).cpu().float().numpy()
        return video

    def prepare_extra_step_kwargs(self, generator, eta):
        accepts_eta = "eta" in set(inspect.signature(self.scheduler.step).parameters.keys())
        extra_step_kwargs = {}
        if accepts_eta: extra_step_kwargs["eta"] = eta
        accepts_generator = "generator" in set(inspect.signature(self.scheduler.step).parameters.keys())
        if accepts_generator: extra_step_kwargs["generator"] = generator
        return extra_step_kwargs

    def check_inputs(self, prompt, height, width, callback_steps):
        if not isinstance(prompt, str) and not isinstance(prompt, list): raise ValueError(f"`prompt` has to be of type `str` or `list` but is {type(prompt)}")
        if height % 8 != 0 or width % 8 != 0: raise ValueError(f"`height` and `width` have to be divisible by 8 but are {height} and {width}.")
        if (callback_steps is None) or (callback_steps is not None and (not isinstance(callback_steps, int) or callback_steps <= 0)): raise ValueError(f"`callback_steps` has to be a positive integer.")

    def update_latents_with_video_length(self, latents, video_length):
        if latents.dim() != 4: raise ValueError(f"Expected latents with 4 dimensions, got {latents.dim()}")
        batch_size, num_channels_latents, height, width = latents.shape
        latents_expanded = latents.unsqueeze(2).expand(batch_size, num_channels_latents, video_length, height, width)
        return latents_expanded

    def prepare_latents(self, batch_size, num_channels_latents, video_length, height, width, dtype, device, generator,latents=None, latents1=None, latents2=None, latents3=None, latents4=None, latents5=None, latents6=None, latents7=None, latents8=None, latents9=None, latents10=None, latents11=None, latents12=None,latents13=None, latents14=None, latents15=None, latents16=None):
        shape = (batch_size, num_channels_latents, video_length, height // self.vae_scale_factor, width // self.vae_scale_factor)
        if isinstance(generator, list) and len(generator) != batch_size: raise ValueError(f"Generators mismatch.")
        if latents1 is  None and latents2 is  None:
            rand_device = "cpu" if device.type == "mps" else device
            if isinstance(generator, list):
                latents = [torch.randn(shape, generator=generator[i], device=rand_device, dtype=dtype) for i in range(batch_size)]
                latents = torch.cat(latents, dim=0).to(device)                
            else:
                latents = torch.randn(shape, generator=generator, device=rand_device, dtype=dtype).to(device)
        else:
            print(f"Latents1 shape: {latents1.shape}")
            print(f"Latents2 shape: {latents2.shape}")
            print(f"Expected shape: {shape}")
            if latents1.shape != shape:
                extended_latents = torch.randn(shape, generator=generator, device=device, dtype=dtype)
                extended_latents[:, :, 0, :, :] = latents1
                extended_latents[:, :, 1, :, :] = latents2
                extended_latents[:, :, 2, :, :] = latents3
                extended_latents[:, :, 3, :, :] = latents4
                extended_latents[:, :, 4, :, :] = latents5
                extended_latents[:, :, 5, :, :] = latents6
                extended_latents[:, :, 6, :, :] = latents7
                extended_latents[:, :, 7, :, :] = latents8
                extended_latents[:, :, 8, :, :] = latents9
                extended_latents[:, :, 9, :, :] = latents10
                extended_latents[:, :, 10, :, :] = latents11
                extended_latents[:, :, 11, :, :] = latents12
                extended_latents[:, :, 12, :, :] = latents13
                extended_latents[:, :, 13, :, :] = latents14
                extended_latents[:, :, 14, :, :] = latents15
                extended_latents[:, :, 15, :, :] = latents16
                latents = extended_latents
                print("update latents = ",latents.shape)
            latents = latents.to(device)

        latents = latents * self.scheduler.init_noise_sigma
        return latents

    @torch.no_grad()
    def __call__(
        self, prompt: Union[str, List[str]], video_length: Optional[int], height: Optional[int] = None, width: Optional[int] = None,
        num_inference_steps: int = 50, guidance_scale: float = 7.5, negative_prompt: Optional[Union[str, List[str]]] = None,
        num_videos_per_prompt: Optional[int] = 1, eta: float = 0.0, generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents1: Optional[torch.FloatTensor] = None, latents2: Optional[torch.FloatTensor] = None, latents3: Optional[torch.FloatTensor] = None, latents4: Optional[torch.FloatTensor] = None,
        latents5: Optional[torch.FloatTensor] = None, latents6: Optional[torch.FloatTensor] = None, latents7: Optional[torch.FloatTensor] = None, latents8: Optional[torch.FloatTensor] = None,
        latents9: Optional[torch.FloatTensor] = None, latents10: Optional[torch.FloatTensor] = None, latents11: Optional[torch.FloatTensor] = None, latents12: Optional[torch.FloatTensor] = None,
        latents13: Optional[torch.FloatTensor] = None, latents14: Optional[torch.FloatTensor] = None, latents15: Optional[torch.FloatTensor] = None, latents16: Optional[torch.FloatTensor] = None,
        latents: Optional[torch.FloatTensor] = None, output_type: Optional[str] = "tensor", return_dict: bool = True, callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
        callback_steps: Optional[int] = 1, watermarking_gamma: float = None, watermarking_delta: float = None, watermarking_mask: Optional[torch.BoolTensor] = None,
        controlnet_images: torch.FloatTensor = None, controlnet_image_index: list = [0], controlnet_conditioning_scale: Union[float, List[float]] = 1.0, **kwargs,
    ):
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor

        self.check_inputs(prompt, height, width, callback_steps)

        batch_size = latents.shape[0] if latents is not None else (len(prompt) if isinstance(prompt, list) else 1)
        device = self._execution_device
        do_classifier_free_guidance = guidance_scale > 1.0

        prompt = prompt if isinstance(prompt, list) else [prompt] * batch_size
        if negative_prompt is not None:
            negative_prompt = negative_prompt if isinstance(negative_prompt, list) else [negative_prompt] * batch_size 
        text_embeddings = self._encode_prompt(prompt, device, num_videos_per_prompt, do_classifier_free_guidance, negative_prompt)

        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        num_channels_latents = self.unet.in_channels
        latents = self.prepare_latents(
            batch_size * num_videos_per_prompt, num_channels_latents, video_length, height, width, text_embeddings.dtype, device, generator,
            latents1=latents1, latents2=latents2, latents3=latents3, latents4=latents4, latents5=latents5, latents6=latents6, latents7=latents7, latents8=latents8,
            latents9=latents9, latents10=latents10, latents11=latents11, latents12=latents12, latents13=latents13, latents14=latents14, latents15=latents15, latents16=latents16,
            latents=latents,
        )
        latents_dtype = latents.dtype
        init_latents = copy.deepcopy(latents)

        if watermarking_gamma is not None:
            watermarking_mask = torch.rand(latents.shape, device=device) < watermarking_gamma
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if watermarking_mask is not None:
                    latents[watermarking_mask] += watermarking_delta * torch.sign(latents[watermarking_mask])
                
                latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                down_block_additional_residuals = mid_block_additional_residual = None
                if (getattr(self, "controlnet", None) != None) and (controlnet_images != None):
                    controlnet_noisy_latents = latent_model_input
                    controlnet_prompt_embeds = text_embeddings
                    controlnet_images = controlnet_images.to(latents.device)
                    controlnet_cond_shape = list(controlnet_images.shape)
                    controlnet_cond_shape[2] = video_length
                    controlnet_cond = torch.zeros(controlnet_cond_shape).to(latents.device)
                    controlnet_conditioning_mask_shape = list(controlnet_cond.shape)
                    controlnet_conditioning_mask_shape[1] = 1
                    controlnet_conditioning_mask = torch.zeros(controlnet_conditioning_mask_shape).to(latents.device)
                    controlnet_cond[:,:,controlnet_image_index] = controlnet_images[:,:,:len(controlnet_image_index)]
                    controlnet_conditioning_mask[:,:,controlnet_image_index] = 1

                    down_block_additional_residuals, mid_block_additional_residual = self.controlnet(
                        controlnet_noisy_latents, t, encoder_hidden_states=controlnet_prompt_embeds, controlnet_cond=controlnet_cond,
                        conditioning_mask=controlnet_conditioning_mask, conditioning_scale=controlnet_conditioning_scale, guess_mode=False, return_dict=False,
                    )

                noise_pred = self.unet(
                    latent_model_input, t, encoder_hidden_states=text_embeddings,
                    down_block_additional_residuals=down_block_additional_residuals, mid_block_additional_residual=mid_block_additional_residual,
                ).sample.to(dtype=latents_dtype)

                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs).prev_sample

                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        callback(i, t, latents)

        video = self.decode_latents(latents)
        if output_type == "tensor": video = torch.from_numpy(video)
        if not return_dict: return video
        return AnimationPipelineOutput(videos=video, init_latents=init_latents)
        
    @torch.inference_mode()
    def get_image_latents(self, image, sample=True, rng_generator=None):
        encoding_dist = self.vae.encode(image).latent_dist
        encoding = encoding_dist.sample(generator=rng_generator) if sample else encoding_dist.mode()
        return encoding * 0.18215
        
    @torch.inference_mode()
    def backward_diffusion(
        self, use_old_emb_i=25, text_embeddings=None, old_text_embeddings=None, new_text_embeddings=None,
        latents: Optional[torch.FloatTensor] = None, num_inference_steps: int = 50, guidance_scale: float = 7.5,
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None, callback_steps: Optional[int] = 1, reverse_process: True = False, **kwargs,
    ):
        do_classifier_free_guidance = guidance_scale > 1.0
        self.scheduler.set_timesteps(num_inference_steps)
        timesteps_tensor = self.scheduler.timesteps.to(self.device)
        latents = latents * self.scheduler.init_noise_sigma
        prompt_to_prompt = old_text_embeddings is not None and new_text_embeddings is not None

        for i, t in enumerate(self.progress_bar(timesteps_tensor if not reverse_process else reversed(timesteps_tensor))):
            if prompt_to_prompt:
                text_embeddings = old_text_embeddings if i < use_old_emb_i else new_text_embeddings

            latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
            latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

            noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample

            if do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

            prev_timestep = t - self.scheduler.config.num_train_timesteps // self.scheduler.num_inference_steps
            if callback is not None and i % callback_steps == 0: callback(i, t, latents)
            
            alpha_prod_t = self.scheduler.alphas_cumprod[t]
            alpha_prod_t_prev = self.scheduler.alphas_cumprod[prev_timestep] if prev_timestep >= 0 else self.scheduler.final_alpha_cumprod
            if reverse_process: alpha_prod_t, alpha_prod_t_prev = alpha_prod_t_prev, alpha_prod_t
            latents = backward_ddim(x_t=latents, alpha_t=alpha_prod_t, alpha_tm1=alpha_prod_t_prev, eps_xt=noise_pred)
        return latents

    def get_text_embedding(self, prompt):
        text_input_ids = self.tokenizer(prompt, padding="max_length", truncation=True, max_length=self.tokenizer.model_max_length, return_tensors="pt").input_ids
        return self.text_encoder(text_input_ids.to(self.device))[0]