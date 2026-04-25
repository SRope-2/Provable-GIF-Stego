"""
run_stego.py — Entry point for Provable GIF Steganography.

Usage examples
--------------
# Sender (hide watermark, requires AnimateDiff + SD weights):
python scripts/run_stego.py --mode hide \\
    --model_path /path/to/stable-diffusion-v1-5 \\
    --animatediff_path /path/to/animatediff \\
    --output_path ./output/ \\
    --prompt "A highly detailed cinematic animation" \\
    --gen_seed 42 \\
    --chacha

# Receiver (extract & verify, requires SD v1-5 only):
python scripts/run_stego.py --mode extract \\
    --model_path /path/to/stable-diffusion-v1-5 \\
    --input_gif_dir ./output/frames/ \\
    --key_dir ./output/keys/ \\
    --watermark_dir ./output/watermarks/ \\
    --group_id 0 \\
    --chacha
"""

import argparse
import os
import time
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
import sys

# Ensure project root is on sys.path regardless of working directory
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.watermark_core import Gaussian_Shading, Gaussian_Shading_chacha
from core.inverse_stable_diffusion import InversableStableDiffusionPipeline
from diffusers import DDIMScheduler

from core.io_utils import *
from core.image_utils import transform_img

device = 'cuda' if torch.cuda.is_available() else 'cpu'


# ---------------------------------------------------------------------------
# Helper: build AnimateDiff pipeline (optional, only needed for hide mode)
# ---------------------------------------------------------------------------
def _load_animation_pipeline(args):
    """
    Load the AnimateDiff AnimationPipeline.

    Requires:
      - AnimateDiff repository on sys.path (args.animatediff_path)
      - Stable Diffusion v1-5 weights (args.model_path)
      - AnimateDiff motion module checkpoint
    """
    if args.animatediff_path:
        sys.path.insert(0, os.path.abspath(args.animatediff_path))

    from core.pipeline_stego import AnimationPipeline
    from animatediff.models.unet import UNet3DConditionModel

    scheduler = DDIMScheduler(
        beta_start=0.00085,
        beta_end=0.012,
        beta_schedule='scaled_linear',
        steps_offset=1,
        clip_sample=False,
    )

    from transformers import CLIPTextModel, CLIPTokenizer
    from diffusers.models import AutoencoderKL

    tokenizer = CLIPTokenizer.from_pretrained(args.model_path, subfolder='tokenizer')
    text_encoder = CLIPTextModel.from_pretrained(args.model_path, subfolder='text_encoder')
    vae = AutoencoderKL.from_pretrained(args.model_path, subfolder='vae')
    unet = UNet3DConditionModel.from_pretrained_2d(
        args.model_path,
        subfolder='unet',
        unet_additional_kwargs={
            'use_motion_module': True,
            'motion_module_resolutions': (1, 2, 4, 8),
            'unet_use_cross_frame_attention': False,
            'unet_use_temporal_attention': False,
            'motion_module_type': 'Vanilla',
            'motion_module_kwargs': {
                'num_attention_heads': 8,
                'num_transformer_block': 1,
                'attention_block_types': ('Temporal_Self', 'Temporal_Self'),
                'temporal_position_encoding': True,
                'temporal_position_encoding_max_len': 32,
                'temporal_attention_dim_div': 1,
            },
        }
    )

    pipe = AnimationPipeline(
        vae=vae, text_encoder=text_encoder, tokenizer=tokenizer,
        unet=unet, scheduler=scheduler,
    ).to(device)

    if args.motion_module_path and os.path.exists(args.motion_module_path):
        state_dict = torch.load(args.motion_module_path, map_location='cpu')
        pipe.unet.load_state_dict(state_dict, strict=False)
        print(f"[*] Loaded motion module: {args.motion_module_path}")

    pipe = pipe.to(torch.float16)
    return pipe


# ---------------------------------------------------------------------------
# Hide mode (Sender)
# ---------------------------------------------------------------------------
def run_hiding(args):
    print(f"\n{'='*55}")
    print(f"[*] MODE: HIDING (Sender)")
    print(f"[*] ENCRYPTION: {'ChaCha20' if args.chacha else 'Standard OTP'}")
    print(f"[*] Group ID   : {args.gen_seed}")
    print(f"{'='*55}\n")

    # 1. Initialise watermark module
    if args.chacha:
        watermark = Gaussian_Shading_chacha(
            args.channel_copy, args.hw_copy, args.fpr, args.user_number
        )
    else:
        watermark = Gaussian_Shading(
            args.channel_copy, args.hw_copy, args.fpr, args.user_number,
            output_dir=args.output_path,   # <-- pass configurable output dir
        )

    # 2. Generate watermark latents
    print("[*] Generating encrypted watermark latents...")
    if args.chacha:
        # Gaussian_Shading_chacha.create_watermark_and_return_w() takes no arguments
        init_latents_w = watermark.create_watermark_and_return_w()
        # chacha variant returns a single tensor; wrap it for uniform handling
        init_latents_list = [init_latents_w] * 16
    else:
        # Gaussian_Shading.create_watermark_and_return_w(group_id, save)
        init_latents_tuple = watermark.create_watermark_and_return_w(
            group_id=args.gen_seed, save=True
        )
        init_latents_list = list(init_latents_tuple)

    print(f"[+] Generated {len(init_latents_list)} per-frame latents.")
    print(f"    Latent shape: {init_latents_list[0].shape}")

    # 3. (Optional) Run AnimateDiff pipeline if weights are available
    if args.model_path and os.path.isdir(args.model_path):
        try:
            print("\n[*] Loading AnimateDiff pipeline...")
            pipe = _load_animation_pipeline(args)
            generator = torch.Generator(device=device).manual_seed(args.gen_seed)
            print(f"[*] Generating GIF with prompt: '{args.prompt}'")
            output = pipe(
                prompt=args.prompt,
                video_length=16,
                height=args.height,
                width=args.width,
                num_inference_steps=args.num_inference_steps,
                guidance_scale=args.guidance_scale,
                generator=generator,
                latents1=init_latents_list[0],  latents2=init_latents_list[1],
                latents3=init_latents_list[2],  latents4=init_latents_list[3],
                latents5=init_latents_list[4],  latents6=init_latents_list[5],
                latents7=init_latents_list[6],  latents8=init_latents_list[7],
                latents9=init_latents_list[8],  latents10=init_latents_list[9],
                latents11=init_latents_list[10], latents12=init_latents_list[11],
                latents13=init_latents_list[12], latents14=init_latents_list[13],
                latents15=init_latents_list[14], latents16=init_latents_list[15],
            )
            frames_dir = os.path.join(args.output_path, f'frames_group{args.gen_seed}')
            os.makedirs(frames_dir, exist_ok=True)
            video = output.videos[0]  # (C, T, H, W) in [0,1]
            from torchvision.utils import save_image
            for t in range(video.shape[1]):
                save_image(video[:, t], os.path.join(frames_dir, f'frame_{t+1:03d}.png'))
            print(f"\n[+] SUCCESS! Frames saved to: {frames_dir}")
        except Exception as e:
            print(f"\n[!] AnimateDiff pipeline not available: {e}")
            print("[!] Watermark latents have been generated and saved.")
            print(f"[+] Keys saved to : {watermark.key_dir}")
            print(f"[+] Watermarks to : {watermark.watermark_dir}")
    else:
        print("\n[!] --model_path not provided or not found; skipping GIF generation.")
        if not args.chacha:
            print(f"[+] Keys saved to      : {os.path.join(args.output_path, 'keys')}")
            print(f"[+] Watermarks saved to: {os.path.join(args.output_path, 'watermarks')}")

    print(f"\n[+] Output directory: {args.output_path}")


# ---------------------------------------------------------------------------
# Extract mode (Receiver)
# ---------------------------------------------------------------------------
def run_extraction(args):
    print(f"\n{'='*55}")
    print(f"[*] MODE: EXTRACTION & VERIFICATION (Receiver)")
    print(f"[*] ENCRYPTION: {'ChaCha20' if args.chacha else 'Standard OTP'}")
    print(f"[*] INPUT DIR  : {args.input_gif_dir}")
    print(f"{'='*55}\n")

    # 1. Load Stable Diffusion inversion pipeline
    print("[*] Loading Inversable Stable Diffusion Pipeline...")
    scheduler = DDIMScheduler(
        beta_start=0.00085,
        beta_end=0.012,
        beta_schedule='scaled_linear',
        steps_offset=1,
        clip_sample=False,
    )
    pipe = InversableStableDiffusionPipeline.from_pretrained(
        args.model_path,
        scheduler=scheduler,
        torch_dtype=torch.float16,
    )
    pipe.safety_checker = None
    pipe = pipe.to(device)

    # 2. Initialise watermark decoder
    if args.chacha:
        watermark = Gaussian_Shading_chacha(
            args.channel_copy, args.hw_copy, args.fpr, args.user_number
        )
    else:
        watermark = Gaussian_Shading(
            args.channel_copy, args.hw_copy, args.fpr, args.user_number,
            output_dir=args.output_path,
        )

    tester_prompt = ''
    text_embeddings = pipe.get_text_embedding(tester_prompt)

    # 3. Per-frame inversion + key XOR
    frames_per_group = 16
    extracted_latents = []

    print("[*] Inverting frames back to latent space...")
    start_time = time.time()

    for frame_id in tqdm(range(frames_per_group), desc="Extracting frames"):
        input_img_path = os.path.join(args.input_gif_dir, f'frame_{frame_id+1:03d}.png')

        if not os.path.exists(input_img_path):
            print(f"[!] Warning: {input_img_path} not found. Skipping...")
            continue

        image_w_distortion = Image.open(input_img_path)
        image_w_distortion = (
            transform_img(image_w_distortion)
            .unsqueeze(0)
            .to(text_embeddings.dtype)
            .to(device)
        )

        image_latents_w = pipe.get_image_latents(image_w_distortion, sample=False)
        reversed_latents_w = pipe.forward_diffusion(
            latents=image_latents_w,
            text_embeddings=text_embeddings,
            guidance_scale=1,
            num_inference_steps=args.num_inversion_steps,
        )

        reversed_m = (reversed_latents_w > 0).int()

        # Load per-frame key (Standard OTP mode only)
        key_file = os.path.join(args.key_dir, f'key{args.group_id}_tensor{frame_id}.pth')
        if os.path.exists(key_file):
            key = torch.load(key_file, map_location=device)
            reversed_sd = (reversed_m + key) % 2
            reversed_watermark = watermark.diffusion_inverse(reversed_sd)
            extracted_latents.append(reversed_watermark.cpu().numpy())
        else:
            print(f"[!] Warning: Missing key file: {key_file}")

    end_time = time.time()

    # 4. Cross-frame majority voting
    if not extracted_latents:
        print("[!] No frames were successfully processed. Aborting.")
        return

    print("\n[*] Applying Cross-Frame Majority Voting...")
    stacked_watermarks = np.stack(extracted_latents, axis=0)
    threshold = len(extracted_latents) / 2.0
    final_watermark_np = (np.sum(stacked_watermarks, axis=0) > threshold).astype(int)
    final_watermark = torch.from_numpy(final_watermark_np).to(device)

    # 5. Accuracy verification
    wm_file = os.path.join(
        args.watermark_dir, f'watermark_tensor{args.group_id}.pth'
    )
    if os.path.exists(wm_file):
        original_watermark = torch.load(wm_file, map_location=device)
        correct_bits = (final_watermark == original_watermark).sum().item()
        total_bits = original_watermark.numel()
        accuracy = correct_bits / total_bits

        print("\n" + "="*55)
        print(f" ACCURACY REPORT — Stego-GIF (Group {args.group_id})")
        print(f" Extraction Time  : {end_time - start_time:.2f} s")
        print(f" Frames processed : {len(extracted_latents)} / {frames_per_group}")
        print(f" Bit Accuracy     : {accuracy * 100:.4f}%")
        print("="*55 + "\n")
    else:
        print(f"[!] Ground-truth watermark not found at: {wm_file}")
        print(f"    Pass --watermark_dir pointing to the saved watermarks directory.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description='Provably Secure and Robust Training-Free GIF Steganography',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Mode
    parser.add_argument('--mode', type=str, required=True, choices=['hide', 'extract'],
                        help='Sender (hide) or Receiver (extract) mode.')

    # Model paths
    parser.add_argument('--model_path', type=str, default='',
                        help='Path to Stable Diffusion v1-5 model directory.')
    parser.add_argument('--animatediff_path', type=str, default='',
                        help='Path to AnimateDiff repository root (added to sys.path).')
    parser.add_argument('--motion_module_path', type=str, default='',
                        help='Path to AnimateDiff motion module .ckpt file.')

    # I/O
    parser.add_argument('--output_path', type=str, default='./output/',
                        help='Root output directory for frames, keys, and watermarks.')
    parser.add_argument('--input_gif_dir', type=str, default='',
                        help='[extract] Directory containing frame_001.png … frame_016.png.')
    parser.add_argument('--key_dir', type=str, default='',
                        help='[extract] Directory containing per-frame key tensors.')
    parser.add_argument('--watermark_dir', type=str, default='',
                        help='[extract] Directory containing ground-truth watermark tensors.')
    parser.add_argument('--group_id', type=int, default=0,
                        help='[extract] Group ID matching the key/watermark file names.')

    # Generation
    parser.add_argument('--prompt', type=str,
                        default='A highly detailed cinematic animation')
    parser.add_argument('--gen_seed', type=int, default=0,
                        help='[hide] Random seed for generation AND group_id for saved files.')
    parser.add_argument('--height', type=int, default=512)
    parser.add_argument('--width', type=int, default=512)
    parser.add_argument('--num_inference_steps', type=int, default=25)
    parser.add_argument('--guidance_scale', type=float, default=7.5)
    parser.add_argument('--num_inversion_steps', type=int, default=25)

    # Watermark cryptographic parameters
    parser.add_argument('--channel_copy', type=int, default=1,
                        help='Channel repetition factor (ch). Watermark ch-dim = 4//ch.')
    parser.add_argument('--hw_copy', type=int, default=2,
                        help='Spatial repetition factor (hw). Watermark spatial = 64//hw.')
    parser.add_argument('--user_number', type=int, default=1_000_000)
    parser.add_argument('--fpr', type=float, default=1e-6,
                        help='Target false positive rate for threshold calibration.')
    parser.add_argument('--chacha', action='store_true',
                        help='Use ChaCha20 stream cipher (Gaussian_Shading_chacha).')

    args = parser.parse_args()

    # Set default key/watermark dirs from output_path when not explicitly given
    if not args.key_dir:
        args.key_dir = os.path.join(args.output_path, 'keys')
    if not args.watermark_dir:
        args.watermark_dir = os.path.join(args.output_path, 'watermarks')

    os.makedirs(args.output_path, exist_ok=True)

    if args.mode == 'hide':
        run_hiding(args)
    elif args.mode == 'extract':
        run_extraction(args)


if __name__ == '__main__':
    main()
