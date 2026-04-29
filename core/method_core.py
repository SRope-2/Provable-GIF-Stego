"""
method_core.py 
"""

import os
import numpy as np
from scipy.stats import norm

# ── ChaCha20 / fallback ────────────────────────────────────────────────────
try:
    from Crypto.Cipher import ChaCha20
    from Crypto.Random import get_random_bytes
    _HAS_CHACHA = True
except ImportError:
    _HAS_CHACHA = False
    import warnings
    warnings.warn(
        "[watermark_core] pycryptodome not found, using numpy PRNG as cipher (TEST ONLY).",
        RuntimeWarning, stacklevel=2
    )


# ══════════════════════════════════════════════════════════════════════════
#  工具函数
# ══════════════════════════════════════════════════════════════════════════

def sign_aware_mapping(bits: np.ndarray) -> np.ndarray:
    """
    保持N(0,1)分布的sign-aware latent mapping。

    z'_{T,j} = s_j * |l|,  l ~ N(0,1)
    s_j = +1 if bit==1 else -1

    可证明 z'_{T,j} ~ N(0,1)，KL散度为0，实现理论不可检测性。

    Parameters
    ----------
    bits : np.ndarray, shape (m,), dtype int, values in {0,1}

    Returns
    -------
    z : np.ndarray, shape (m,), float32 — 符合N(0,1)的latent向量
    """
    l = np.abs(np.random.randn(len(bits)).astype(np.float32))   # |l|
    signs = np.where(bits == 1, 1.0, -1.0).astype(np.float32)  # s_j
    return signs * l   # z'_{T,j}


def inverse_sign_mapping(z: np.ndarray) -> np.ndarray:
    """
    从latent恢复bit。

    b_j = 0 if sign(z_j) == -1 else 1
    """
    return (z > 0).astype(np.int32)


def preprocess_secret(S: np.ndarray, d: int) -> np.ndarray:
    """
    将secret S重复d次，构建冗余序列S^d。

    Parameters
    ----------
    S : np.ndarray, shape (m,), dtype int, values in {0,1}
    d : int — 重复次数

    Returns
    -------
    S_d : np.ndarray, shape (m*d,)
    """
    return np.tile(S, d)


def intra_frame_vote(S_d_recovered: np.ndarray, d: int) -> np.ndarray:
    """
    将S^d_i (m*d个bit)按d次重复聚合，恢复单帧secret S_i。

    Parameters
    ----------
    S_d_recovered : np.ndarray, shape (m*d,)
    d : int

    Returns
    -------
    S_i : np.ndarray, shape (m,) — 单帧恢复结果
    """
    m = len(S_d_recovered) // d
    chunks = S_d_recovered[:m * d].reshape(d, m)  # (d, m)
    votes = chunks.sum(axis=0)                     # 对d次重复求和
    return (votes > d / 2).astype(np.int32)        # majority vote


def inter_frame_vote(frame_secrets: list) -> np.ndarray:
    """

    Parameters
    ----------
    frame_secrets : list of np.ndarray, each shape (m,)

    Returns
    -------
    S_final : np.ndarray, shape (m,)
    """
    stacked = np.stack(frame_secrets, axis=0)      # (n, m)
    votes = stacked.sum(axis=0)                    # 按位置求和
    n = len(frame_secrets)
    return (votes > n / 2).astype(np.int32)


# ══════════════════════════════════════════════════════════════════════════
#  ChaCha20 加密/解密
# ══════════════════════════════════════════════════════════════════════════

class StreamCipher:
    """
    对单帧secret进行加密/解密。
    - 有pycryptodome时使用ChaCha20 (论文设计)
    - 无pycryptodome时使用numpy PRNG XOR (仅测试)
    """

    def __init__(self):
        self.key = None
        self.nonce = None
        self._seed = None  # fallback用

    def generate_key(self):
        """生成一次性密钥（每帧独立调用）。"""
        if _HAS_CHACHA:
            self.key = get_random_bytes(32)
            self.nonce = get_random_bytes(12)
        else:
            self._seed = int.from_bytes(os.urandom(4), 'big')

    def encrypt(self, bits: np.ndarray) -> np.ndarray:
        """bits: {0,1} array → 加密后的bits。"""
        if _HAS_CHACHA:
            raw = np.packbits(bits).tobytes()
            cipher = ChaCha20.new(key=self.key, nonce=self.nonce)
            enc = np.unpackbits(np.frombuffer(cipher.encrypt(raw), dtype=np.uint8))
        else:
            rng = np.random.RandomState(self._seed)
            keystream = rng.randint(0, 2, size=len(bits)).astype(np.int32)
            enc = (bits ^ keystream).astype(np.int32)
        return enc[:len(bits)]

    def decrypt(self, bits: np.ndarray) -> np.ndarray:
        """ChaCha20是对称的；fallback XOR同理。"""
        if _HAS_CHACHA:
            raw = np.packbits(bits).tobytes()
            cipher = ChaCha20.new(key=self.key, nonce=self.nonce)
            dec = np.unpackbits(np.frombuffer(cipher.decrypt(raw), dtype=np.uint8))
        else:
            rng = np.random.RandomState(self._seed)
            keystream = rng.randint(0, 2, size=len(bits)).astype(np.int32)
            dec = (bits ^ keystream).astype(np.int32)
        return dec[:len(bits)]

    def state_dict(self) -> dict:
        """保存密钥状态（供receiver使用）。"""
        if _HAS_CHACHA:
            return {'key': self.key, 'nonce': self.nonce}
        return {'seed': self._seed}

    def load_state_dict(self, d: dict):
        """加载密钥状态。"""
        if _HAS_CHACHA:
            self.key = d['key']
            self.nonce = d['nonce']
        else:
            self._seed = d['seed']


# ══════════════════════════════════════════════════════════════════════════
#  GIFStegoEmbedder — 对应论文 Embedding Module
# ══════════════════════════════════════════════════════════════════════════

class GIFStegoEmbedder:
    """
    论文 Section IV-B: Embedding Module。

    用法:
        embedder = GIFStegoEmbedder(n_frames=16, d=16, latent_dim=4*64*64)
        latents, keys = embedder.embed(S)
        # latents: list of n np.ndarray, 每个shape (latent_dim,)
        # keys:    list of n StreamCipher，供receiver使用
    """

    def __init__(self, n_frames: int = 16, d: int = 16, latent_dim: int = 4 * 64 * 64):
        """
        Parameters
        ----------
        n_frames : int — GIF帧数 (n=16)
        d        : int — secret重复次数 (默认d=16)
        latent_dim : int — 每帧latent维度 (4*64*64=16384)
        """
        self.n = n_frames
        self.d = d
        self.latent_dim = latent_dim
        # 论文公式: C_frame = latent_dim / d
        self.capacity = latent_dim // d  # 每帧能嵌入的secret bit数

    def embed(self, S: np.ndarray):
        """
        完整嵌入流程:
          S -> 重复d次 -> 每帧独立ChaCha20加密 -> sign-aware mapping -> latent Z'_T

        Parameters
        ----------
        S : np.ndarray, shape (capacity,), dtype int, values in {0,1}
            用户的secret message

        Returns
        -------
        latents : list of np.ndarray  — 每帧的初始latent，长度n
        ciphers : list of StreamCipher — 每帧的密钥，供receiver持有
        """
        assert len(S) == self.capacity, (
            f"Secret长度应为{self.capacity}bits，实际{len(S)}bits\n"
            f"(由 latent_dim={self.latent_dim} / d={self.d} 决定)"
        )

        # Step 1: 重复d次 → S^d，shape (latent_dim,)
        S_d = preprocess_secret(S, self.d)  # (latent_dim,)

        latents = []
        ciphers = []

        for i in range(self.n):
            # Step 2: 每帧独立密钥加密 → m_i，shape (latent_dim,)
            cipher = StreamCipher()
            cipher.generate_key()
            m_i = cipher.encrypt(S_d)

            # Step 3: sign-aware mapping → Z'_T，shape (latent_dim,)  [论文公式2,3]
            z_i = sign_aware_mapping(m_i)

            latents.append(z_i.reshape(-1))  
            ciphers.append(cipher)

        return latents, ciphers


class GIFStegoExtractor:
    """
    论文 Section IV-C & IV-D: Extraction Module + Verification Module。

    用法:
        extractor = GIFStegoExtractor(n_frames=16, d=16, latent_dim=4*64*64)
        S_recovered = extractor.extract(inverted_latents, ciphers)
    """

    def __init__(self, n_frames: int = 16, d: int = 16, latent_dim: int = 4 * 64 * 64):
        self.n = n_frames
        self.d = d
        self.latent_dim = latent_dim
        self.capacity = latent_dim // d

    def extract_single_frame(self, z_recovered: np.ndarray, cipher: StreamCipher) -> np.ndarray:
        """
        单帧提取流程 (Extraction Module):
          Z'_T → 取符号恢复m_i → ChaCha20解密 → 帧内majority vote → S_i

        Parameters
        ----------
        z_recovered : np.ndarray, shape (4,64,64) or (latent_dim,)
            DDIM inversion后恢复的latent

        Returns
        -------
        S_i : np.ndarray, shape (capacity,) — 单帧恢复的secret
        """
        z_flat = z_recovered.flatten()[:self.latent_dim]

        m_i_recovered = inverse_sign_mapping(z_flat)  # shape (latent_dim,)

        S_d_recovered = cipher.decrypt(m_i_recovered)

        S_i = intra_frame_vote(S_d_recovered, self.d)

        return S_i

    def extract(self, inverted_latents: list, ciphers: list) -> np.ndarray:
        """
        完整提取流程 (Extraction + Verification Module):
          各帧单独提取 → 帧间majority voting → S'

        Parameters
        ----------
        inverted_latents : list of np.ndarray — DDIM inversion结果，长度n
        ciphers          : list of StreamCipher — 对应每帧密钥，长度n

        Returns
        -------
        S_final : np.ndarray, shape (capacity,) — 最终恢复的secret
        """
        assert len(inverted_latents) == len(ciphers) == self.n

        # Extraction Module: 每帧独立提取
        frame_secrets = []
        for i, (z, cipher) in enumerate(zip(inverted_latents, ciphers)):
            S_i = self.extract_single_frame(z, cipher)
            frame_secrets.append(S_i)

        # Verification Module: 帧间majority voting [论文公式(8), Fig.3]
        S_final = inter_frame_vote(frame_secrets)
        return S_final




class Gaussian_Shading:

    def __init__(self, ch_factor, hw_factor, fpr, user_number, output_dir='./output'):
        self.ch = ch_factor
        self.hw = hw_factor
        self.output_dir = output_dir
        self.watermark_dir = os.path.join(output_dir, 'watermarks')
        self.key_dir = os.path.join(output_dir, 'keys')
        self.latent_dim = 4 * 64 * 64
        self.d = self.ch * self.hw * self.hw         
        self.capacity = self.latent_dim // self.d
        self._ciphers = []
        self._watermark = None

        # 复用新实现
        self._embedder = GIFStegoEmbedder(n_frames=16, d=self.d,
                                           latent_dim=self.latent_dim)
        self._extractor = GIFStegoExtractor(n_frames=16, d=self.d,
                                             latent_dim=self.latent_dim)

    def create_watermark_and_return_w(self, group_id=0, save=True, secret=None):
        """
        生成secret并嵌入，返回16个latent。
        secret: 用户传入的bit数组(shape=(capacity,))；None时随机生成。
        """
        if secret is None:
            secret = np.random.randint(0, 2, self.capacity).astype(np.int32)
        self._watermark = secret

        if save:
            os.makedirs(self.watermark_dir, exist_ok=True)
            np.save(os.path.join(self.watermark_dir,
                                 f'watermark_{group_id}.npy'), secret)

        latents, ciphers = self._embedder.embed(secret)
        self._ciphers = ciphers
        latents_4d = [z.reshape(4, 64, 64) if z.size == 4*64*64 else z
                      for z in latents]

        if save:
            os.makedirs(self.key_dir, exist_ok=True)
            for j, c in enumerate(ciphers):
                np.save(os.path.join(self.key_dir,
                                     f'key_{group_id}_frame{j}.npy'),
                        np.array(list(c.state_dict().values()), dtype=object),
                        allow_pickle=True)
        return tuple(latents_4d)

    def eval_watermark(self, inverted_latents, group_id=0):
        S_recovered = self._extractor.extract(inverted_latents, self._ciphers)
        acc = (S_recovered == self._watermark).mean()
        return float(acc)
