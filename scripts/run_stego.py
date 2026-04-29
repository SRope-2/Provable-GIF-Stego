"""
run_stego.py — Entry point for Provable GIF Steganography.
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

from core.watermark_core import Gaussian_Shading, StreamCipher
from core.inverse_stable_diffusion import InversableStableDiffusionPipeline
from diffusers import DDIMScheduler

from core.io_utils import *
# 引入图像处理与失真攻击模块
from core.image_utils import transform_img, image_distortion

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# ---------------------------------------------------------------------------
# Helper: build AnimateDiff pipeline (optional, only needed for hide mode)
# ---------------------------------------------------------------------------
def _load_animation_pipeline(args):
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
    print(f"[*] Group ID   : {args.gen_seed}")
    print(f"{'='*55}\n")

    # 1. 初始化水印模块
    watermark = Gaussian_Shading(
        args.channel_copy, args.hw_copy, args.fpr, args.user_number,
        output_dir=args.output_path, 
    )

    # 2. 生成每帧独立的加密隐变量
    print("[*] Generating encrypted watermark latents...")
    init_latents_tuple = watermark.create_watermark_and_return_w(
        group_id=args.gen_seed, save=True
    )
    init_latents_list = list(init_latents_tuple)

    print(f"[+] Generated {len(init_latents_list)} per-frame latents.")
    print(f"    Latent shape: {init_latents_list[0].shape}")

    # 3. 运行 AnimateDiff 
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
            video = output.videos[0]  
            from torchvision.utils import save_image
            for t in range(video.shape[1]):
                save_image(video[:, t], os.path.join(frames_dir, f'frame_{t+1:03d}.png'))
            print(f"\n[+] SUCCESS! Frames saved to: {frames_dir}")
        except Exception as e:
            print(f"\n[!] AnimateDiff pipeline error: {e}")
            print("[!] Watermark latents have been generated and saved.")
    else:
        print("\n[!] --model_path not provided or not found; skipping GIF generation.")

    print(f"[+] Keys saved to      : {watermark.key_dir}")
    print(f"[+] Watermarks saved to: {watermark.watermark_dir}")
    print(f"\n[+] Output directory: {args.output_path}")


# ---------------------------------------------------------------------------
# Extract mode (Receiver)
# ---------------------------------------------------------------------------
def run_extraction(args):
    print(f"\n{'='*55}")
    print(f"[*] MODE: EXTRACTION & VERIFICATION (Receiver)")
    print(f"[*] INPUT DIR  : {args.input_gif_dir}")
    print(f"{'='*55}\n")

    # 1. 载入反演模型
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

    watermark = Gaussian_Shading(
        args.channel_copy, args.hw_copy, args.fpr, args.user_number,
        output_dir=args.output_path,
    )

    tester_prompt = ''
    text_embeddings = pipe.get_text_embedding(tester_prompt)

    frames_per_group = 16
    extracted_latents = []
    loaded_ciphers = []

    print("[*] Inverting frames back to latent space...")
    start_time = time.time()

    for frame_id in tqdm(range(frames_per_group), desc="Extracting frames"):
        input_img_path = os.path.join(args.input_gif_dir, f'frame_{frame_id+1:03d}.png')

        if not os.path.exists(input_img_path):
            print(f"[!] Warning: {input_img_path} not found. Skipping...")
            continue

        image_w_distortion = Image.open(input_img_path)

        # ====== 核心新增：应用信道失真模拟 (Lossy Channel) ======
        image_w_distortion = image_distortion(image_w_distortion, seed=args.group_id, args=args)
        # ==========================================================

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

        extracted_latents.append(reversed_latents_w.cpu().numpy())

        # 读取 numpy 格式的密钥
        key_file = os.path.join(args.key_dir, f'key_{args.group_id}_frame{frame_id}.npy')
        if os.path.exists(key_file):
            key_data = np.load(key_file, allow_pickle=True)
            cipher = StreamCipher()
            if len(key_data) == 2:
                cipher.load_state_dict({'key': key_data[0], 'nonce': key_data[1]})
            else:
                cipher.load_state_dict({'seed': key_data[0]})
            loaded_ciphers.append(cipher)
        else:
            print(f"[!] Warning: Missing key file: {key_file}")

    end_time = time.time()

    # 调用底层 API 自动完成解密与多重多数投票
    if len(extracted_latents) > 0 and len(loaded_ciphers) == len(extracted_latents):
        print("\n[*] Applying Extraction and Majority Voting...")
        S_recovered = watermark._extractor.extract(extracted_latents, loaded_ciphers)
        
        # 读取 Ground-Truth 验证准确率
        wm_file = os.path.join(args.watermark_dir, f'watermark_{args.group_id}.npy')
        if os.path.exists(wm_file):
            original_watermark = np.load(wm_file)
            correct_bits = (S_recovered == original_watermark).sum()
            total_bits = original_watermark.size
            accuracy = correct_bits / total_bits

            print("\n" + "="*55)
            print(f" ACCURACY REPORT — Stego-GIF (Group {args.group_id})")
            print(f" Extraction Time  : {end_time - start_time:.2f} s")
            print(f" Frames processed : {len(extracted_latents)} / {frames_per_group}")
            print(f" Bit Accuracy     : {accuracy * 100:.4f}%")
            print("="*55 + "\n")
        else:
            print(f"[!] Ground-truth watermark not found at: {wm_file}")
    else:
        print("[!] Key extraction failed or missing frames. Cannot verify.")

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description='Provably Secure and Robust Training-Free GIF Steganography',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument('--mode', type=str, required=True, choices=['hide', 'extract'], help='Sender (hide) or Receiver (extract) mode.')
    parser.add_argument('--model_path', type=str, default='', help='Path to Stable Diffusion v1-5 model directory.')
    parser.add_argument('--animatediff_path', type=str, default='', help='Path to AnimateDiff repository root.')
    parser.add_argument('--motion_module_path', type=str, default='', help='Path to AnimateDiff motion module .ckpt file.')
    parser.add_argument('--output_path', type=str, default='./output/', help='Root output directory.')
    parser.add_argument('--input_gif_dir', type=str, default='', help='[extract] Directory containing frame_001.png … frame_016.png.')
    parser.add_argument('--key_dir', type=str, default='', help='[extract] Directory containing per-frame key tensors.')
    parser.add_argument('--watermark_dir', type=str, default='', help='[extract] Directory containing ground-truth watermark tensors.')
    parser.add_argument('--group_id', type=int, default=0, help='[extract] Group ID.')
    parser.add_argument('--prompt', type=str, default='A highly detailed cinematic animation')
    parser.add_argument('--gen_seed', type=int, default=0, help='[hide] Random seed and group_id.')
    parser.add_argument('--height', type=int, default=512)
    parser.add_argument('--width', type=int, default=512)
    parser.add_argument('--num_inference_steps', type=int, default=25)
    parser.add_argument('--guidance_scale', type=float, default=7.5)
    parser.add_argument('--num_inversion_steps', type=int, default=25)
    parser.add_argument('--channel_copy', type=int, default=1, help='Channel repetition factor (ch).')
    parser.add_argument('--hw_copy', type=int, default=2, help='Spatial repetition factor (hw).')
    parser.add_argument('--user_number', type=int, default=1_000_000)
    parser.add_argument('--fpr', type=float, default=1e-6)
    parser.add_argument('--chacha', action='store_true', help='Use ChaCha20 stream cipher.')

    # ===== 新增：鲁棒性测试 (Robustness Evaluation) 命令行参数 =====
    parser.add_argument('--jpeg_ratio', type=int, default=None, help='JPEG compression quality (e.g., 50)')
    parser.add_argument('--random_crop_ratio', type=float, default=None)
    parser.add_argument('--random_drop_ratio', type=float, default=None)
    parser.add_argument('--resize_ratio', type=float, default=None, help='Resize ratio (e.g., 0.5)')
    parser.add_argument('--gaussian_blur_r', type=float, default=None, help='Gaussian blur radius (e.g., 3)')
    parser.add_argument('--median_blur_k', type=int, default=None, help='Median blur kernel size')
    parser.add_argument('--gaussian_std', type=float, default=None, help='Gaussian noise standard deviation')
    parser.add_argument('--sp_prob', type=float, default=None, help='Salt and pepper noise probability')
    parser.add_argument('--brightness_factor', type=float, default=None)

    args = parser.parse_args()

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
