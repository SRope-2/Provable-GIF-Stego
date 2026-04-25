import os
import torch
from scipy.stats import norm, truncnorm
from functools import reduce
from scipy.special import betainc
import numpy as np
from Crypto.Cipher import ChaCha20
from Crypto.Random import get_random_bytes
from reedsolo import RSCodec
import torch.nn as nn

class Gaussian_Shading_chacha:
    """
    Single-frame Gaussian Shading watermark with ChaCha20 stream cipher (Sender side).

    The watermark w is encrypted via ChaCha20 before truncated-normal sampling,
    providing cryptographic security guarantees (IND-CPA hardness).
    Used in ChaCha20 mode (--chacha flag).
    """
    def __init__(self, ch_factor, hw_factor, fpr, user_number):
        self.ch = ch_factor
        self.hw = hw_factor
        self.nonce = None
        self.key = None
        self.watermark = None
        self.latentlength = 4 * 64 * 64
        self.marklength = self.latentlength // (self.ch * self.hw * self.hw)
        self._device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.threshold = 1 if self.hw == 1 and self.ch == 1 else self.ch * self.hw * self.hw // 2
        self.tp_onebit_count = 0
        self.tp_bits_count = 0
        self.tau_onebit = None
        self.tau_bits = None

        for i in range(self.marklength):
            fpr_onebit = betainc(i + 1, self.marklength - i, 0.5)
            fpr_bits = betainc(i + 1, self.marklength - i, 0.5) * user_number
            if fpr_onebit <= fpr and self.tau_onebit is None:
                self.tau_onebit = i / self.marklength
            if fpr_bits <= fpr and self.tau_bits is None:
                self.tau_bits = i / self.marklength

    def stream_key_encrypt(self, sd):
        self.key = get_random_bytes(32)
        self.nonce = get_random_bytes(12)
        cipher = ChaCha20.new(key=self.key, nonce=self.nonce)
        m_byte = cipher.encrypt(np.packbits(sd).tobytes())
        m_bit = np.unpackbits(np.frombuffer(m_byte, dtype=np.uint8))
        return m_bit

    def truncSampling(self, message):
        z = np.zeros(self.latentlength)
        denominator = 2.0
        ppf = [norm.ppf(j / denominator) for j in range(int(denominator) + 1)]
        for i in range(self.latentlength):
            dec_mes = reduce(lambda a, b: 2 * a + b, message[i: i + 1])
            dec_mes = int(dec_mes)
            z[i] = truncnorm.rvs(ppf[dec_mes], ppf[dec_mes + 1])
        z = torch.from_numpy(z).reshape(1, 4, 64, 64).float()
        return z.to(self._device)

    def create_watermark_and_return_w(self):
        """Generate watermark and return a single encrypted initial latent."""
        self.watermark = torch.randint(
            0, 2, [1, 4 // self.ch, 64 // self.hw, 64 // self.hw]
        ).to(self._device)
        sd = self.watermark.repeat(1, self.ch, self.hw, self.hw)
        m = self.stream_key_encrypt(sd.flatten().cpu().numpy())
        w = self.truncSampling(m)
        return w

    def stream_key_decrypt(self, reversed_m):
        cipher = ChaCha20.new(key=self.key, nonce=self.nonce)
        sd_byte = cipher.decrypt(np.packbits(reversed_m).tobytes())
        sd_bit = np.unpackbits(np.frombuffer(sd_byte, dtype=np.uint8))
        sd_tensor = torch.from_numpy(sd_bit).reshape(1, 4, 64, 64).to(torch.uint8)
        return sd_tensor.to(self._device)

    def diffusion_inverse(self,watermark_r):
        ch_stride = 4 // self.ch
        hw_stride = 64 // self.hw
        ch_list = [ch_stride] * self.ch
        hw_list = [hw_stride] * self.hw
        split_dim1 = torch.cat(torch.split(watermark_r, tuple(ch_list), dim=1), dim=0)
        split_dim2 = torch.cat(torch.split(split_dim1, tuple(hw_list), dim=2), dim=0)
        split_dim3 = torch.cat(torch.split(split_dim2, tuple(hw_list), dim=3), dim=0)
        vote = torch.sum(split_dim3, dim=0).clone()
        vote[vote <= self.threshold] = 0
        vote[vote > self.threshold] = 1
        return vote

    def eval_watermark(self, reversed_w):
        reversed_m = (reversed_w > 0).int()
        reversed_sd = self.stream_key_decrypt(reversed_m.flatten().cpu().numpy())
        reversed_watermark = self.diffusion_inverse(reversed_sd)
        correct = (reversed_watermark == self.watermark).float().mean().item()
        if correct >= self.tau_onebit:
            self.tp_onebit_count = self.tp_onebit_count+1
        if correct >= self.tau_bits:
            self.tp_bits_count = self.tp_bits_count + 1
        return correct

    def get_tpr(self):
        return self.tp_onebit_count, self.tp_bits_count


class Gaussian_Shading:
    """
    Multi-frame Gaussian Shading watermark for GIF steganography (Sender side).

    This class embeds a shared watermark across all 16 frames of a GIF by:
      1. Generating a compact binary watermark `w` (shape [1, C/ch, H/hw, W/hw]).
      2. For each frame j (0..15), sampling an independent one-time-pad key `key_j`.
      3. Computing per-frame message `m_j = (sd XOR key_j)` and converting it via
         truncated-normal sampling into an initial latent `z_j` injected into AnimateDiff.

    At extraction, the Receiver inverts each frame to its latent, XORs with the stored
    key, and applies majority voting across all 16 frames to recover `w`.

    Parameters
    ----------
    ch_factor : int
        Channel repetition factor (ch). Watermark channel dim = 4 // ch.
    hw_factor : int
        Spatial repetition factor (hw). Watermark spatial dim = 64 // hw.
    fpr : float
        Target false positive rate for threshold calibration.
    user_number : int
        User population size, used for traceability threshold τ_bits.
    output_dir : str, optional
        Root directory for saving watermarks and keys (default: './output').
    """
    def __init__(self, ch_factor, hw_factor, fpr, user_number, output_dir='./output'):
        self.ch = ch_factor
        self.hw = hw_factor
        self.key = None
        self.watermark = None
        self.latentlength = 4 * 64 * 64
        self.marklength = self.latentlength // (self.ch * self.hw * self.hw)
        self.output_dir = output_dir

        # Save directories
        self.watermark_dir = os.path.join(output_dir, 'watermarks')
        self.key_dir = os.path.join(output_dir, 'keys')

        self.threshold = 1 if self.hw == 1 and self.ch == 1 else self.ch * self.hw * self.hw // 2
        self.tp_onebit_count = 0
        self.tp_bits_count = 0
        self.tau_onebit = None
        self.tau_bits = None

        for i in range(self.marklength):
            fpr_onebit = betainc(i + 1, self.marklength - i, 0.5)
            fpr_bits = betainc(i + 1, self.marklength - i, 0.5) * user_number
            if fpr_onebit <= fpr and self.tau_onebit is None:
                self.tau_onebit = i / self.marklength
            if fpr_bits <= fpr and self.tau_bits is None:
                self.tau_bits = i / self.marklength

    def truncSampling(self, message):
        z = np.zeros(self.latentlength)
        denominator = 2.0
        ppf = [norm.ppf(j / denominator) for j in range(int(denominator) + 1)]
        for i in range(self.latentlength):
            dec_mes = reduce(lambda a, b: 2 * a + b, message[i: i + 1])
            dec_mes = int(dec_mes)
            z[i] = truncnorm.rvs(ppf[dec_mes], ppf[dec_mes + 1])
        z = torch.from_numpy(z).reshape(1, 4, 64, 64).half()
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        return z.to(device)

    def create_watermark_and_return_w(self, group_id=0, save=True):
        """
        Generate a binary watermark and 16 per-frame initial latents.

        Parameters
        ----------
        group_id : int
            Index used for naming saved files (watermark_tensor{group_id}.pth,
            key{group_id}_tensor{j}.pth). Default 0.
        save : bool
            Whether to persist watermark and key tensors to disk. Default True.

        Returns
        -------
        tuple of 16 torch.FloatTensor
            Per-frame initial latents z_0 ... z_15, each of shape [1, 4, 64, 64].
        """
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.watermark = torch.randint(
            0, 2, [1, 4 // self.ch, 64 // self.hw, 64 // self.hw]
        ).to(device)

        if save:
            os.makedirs(self.watermark_dir, exist_ok=True)
            torch.save(
                self.watermark.cpu(),
                os.path.join(self.watermark_dir, f'watermark_tensor{group_id}.pth')
            )

        sd = self.watermark.repeat(1, self.ch, self.hw, self.hw)
        w_list = []
        self.key = []

        if save:
            os.makedirs(self.key_dir, exist_ok=True)

        for j in range(16):
            key_j = torch.randint(0, 2, [1, 4, 64, 64]).to(device)
            self.key.append(key_j)
            if save:
                torch.save(
                    key_j.cpu(),
                    os.path.join(self.key_dir, f'key{group_id}_tensor{j}.pth')
                )
            m = ((sd + key_j) % 2).flatten().cpu().numpy()
            current_w = self.truncSampling(m)
            w_list.append(current_w)

        return tuple(w_list)

    def diffusion_inverse(self, watermark_sd):
        ch_stride = 4 // self.ch
        hw_stride = 64 // self.hw
        ch_list = [ch_stride] * self.ch
        hw_list = [hw_stride] * self.hw
        split_dim1 = torch.cat(torch.split(watermark_sd, tuple(ch_list), dim=1), dim=0)
        split_dim2 = torch.cat(torch.split(split_dim1, tuple(hw_list), dim=2), dim=0)
        split_dim3 = torch.cat(torch.split(split_dim2, tuple(hw_list), dim=3), dim=0)
        vote = torch.sum(split_dim3, dim=0).clone()
        vote[vote <= self.threshold] = 0
        vote[vote > self.threshold] = 1
        return vote

    def eval_watermark(self, reversed_watermark, group_id=0, watermark_path=None):
        """
        Evaluate bit accuracy of the recovered watermark.

        Parameters
        ----------
        reversed_watermark : torch.Tensor
            The decoded watermark tensor recovered by the Receiver.
        group_id : int
            Index used to locate the saved ground-truth watermark file.
        watermark_path : str, optional
            Explicit path to the ground-truth .pth file. If None, the path is
            inferred from self.watermark_dir and group_id.

        Returns
        -------
        float
            Bit accuracy in [0, 1].
        """
        if watermark_path is not None:
            wm_file = watermark_path
        else:
            # Try the standard naming used by create_watermark_and_return_w
            wm_file = os.path.join(self.watermark_dir, f'watermark_tensor{group_id}.pth')

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.watermark = torch.load(wm_file, map_location=device)
        print("Target Watermark (self.watermark):")
        print(self.watermark.cpu().numpy())

        correct = (reversed_watermark == self.watermark).float().mean().item()
        if correct >= self.tau_onebit:
            self.tp_onebit_count += 1
        if correct >= self.tau_bits:
            self.tp_bits_count += 1
        return correct

    def get_tpr(self):
        return self.tp_onebit_count, self.tp_bits_count

class TrainableWatermarkGenerator(nn.Module):
    def __init__(self, ch=8, hw=8):
        super().__init__()
        self.ch = ch
        self.hw = hw
        
        self.key_gen = nn.Sequential(
            nn.Conv2d(4, 16, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(16, 4, 1),
            nn.Sigmoid()
        )
        
        self.wm_gen = nn.Sequential(
            nn.Conv2d(4//ch, 16, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(16, 4//ch, 1),
            nn.Sigmoid()
        )

    def forward(self, base_latents):
        dynamic_key = self.key_gen(base_latents)
        core_wm = self.wm_gen(base_latents[:, :4//self.ch])
        sd = core_wm.repeat(1,1,self.hw,self.hw)
        sd = sd.repeat(1,self.ch,1,1)
        return base_latents + (sd + dynamic_key) % 2

class WatermarkExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.Conv2d(4, 16, 3, padding=1),
            nn.GroupNorm(4, 16),
            nn.GELU(),
            nn.Conv2d(16, 1, 1),
            nn.Sigmoid()
        )
        
    def forward(self, noisy_latents):
        return self.decoder(noisy_latents)