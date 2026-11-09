"""GPU data pipeline for CIFAR.

Kaggle notebooks only have 4 vCPUs, which cannot feed two T4s with the PIL
AutoAugment pipeline. This module keeps the whole dataset on the GPU as uint8
and reproduces the training transforms of finetune.py batch-wise:

    RandomHorizontalFlip -> RandomCrop(32, padding=4) -> CIFAR10Policy (AutoAugment)
    -> ToTensor -> Normalize -> RandomErasing

The AutoAugment ops follow utils/transforms.py (PIL semantics: fill colour 128,
same magnitudes/probabilities/random signs, uint8 rounding after every op).
All random decisions are drawn on the CPU with numpy, so no op forces a
GPU->CPU synchronisation.
"""
import math

import numpy as np
import torch
import torch.nn.functional as F

# --------------------------------------------------------------------------- #
#  AutoAugment CIFAR10 policy (identical to utils/autoaug.py::CIFAR10Policy)   #
# --------------------------------------------------------------------------- #
CIFAR10_POLICY = [
    (0.1, "invert", 7, 0.2, "contrast", 6),
    (0.7, "rotate", 2, 0.3, "translateX", 9),
    (0.8, "sharpness", 1, 0.9, "sharpness", 3),
    (0.5, "shearY", 8, 0.7, "translateY", 9),
    (0.5, "autocontrast", 8, 0.9, "equalize", 2),
    (0.2, "shearY", 7, 0.3, "posterize", 7),
    (0.4, "color", 3, 0.6, "brightness", 7),
    (0.3, "sharpness", 9, 0.7, "brightness", 9),
    (0.6, "equalize", 5, 0.5, "equalize", 1),
    (0.6, "contrast", 7, 0.6, "sharpness", 5),
    (0.7, "color", 7, 0.5, "translateX", 8),
    (0.3, "equalize", 7, 0.4, "autocontrast", 8),
    (0.4, "translateY", 3, 0.2, "sharpness", 6),
    (0.9, "brightness", 6, 0.2, "color", 8),
    (0.5, "solarize", 2, 0.0, "invert", 3),
    (0.2, "equalize", 0, 0.6, "autocontrast", 0),
    (0.2, "equalize", 8, 0.6, "equalize", 4),
    (0.9, "color", 9, 0.6, "equalize", 6),
    (0.8, "autocontrast", 4, 0.2, "solarize", 8),
    (0.1, "brightness", 3, 0.7, "color", 0),
    (0.4, "solarize", 5, 0.9, "autocontrast", 3),
    (0.9, "translateY", 9, 0.7, "translateY", 9),
    (0.9, "autocontrast", 2, 0.8, "solarize", 3),
    (0.8, "equalize", 8, 0.1, "invert", 3),
    (0.7, "translateY", 9, 0.9, "autocontrast", 1),
]

RANGES = {
    "shearX": np.linspace(0, 0.3, 10),
    "shearY": np.linspace(0, 0.3, 10),
    "translateX": np.linspace(0, 150 / 331, 10),
    "translateY": np.linspace(0, 150 / 331, 10),
    "rotate": np.linspace(0, 30, 10),
    "color": np.linspace(0.0, 0.9, 10),
    "posterize": np.round(np.linspace(8, 4, 10), 0).astype(np.int32),
    "solarize": np.linspace(256, 0, 10),
    "contrast": np.linspace(0.0, 0.9, 10),
    "sharpness": np.linspace(0.0, 0.9, 10),
    "brightness": np.linspace(0.0, 0.9, 10),
    "autocontrast": [0] * 10,
    "equalize": [0] * 10,
    "invert": [0] * 10,
}
OP_NAMES = list(RANGES)
# ops whose magnitude gets a random sign in utils/transforms.py
SIGNED = {"shearX", "shearY", "translateX", "translateY", "color", "contrast", "sharpness", "brightness"}
FILL = 128.0


def _q(x):
    """Emulate the uint8 round-trip PIL does after every op."""
    return x.round_().clamp_(0, 255)


def _affine(x, mats, mode):
    """PIL-style affine: mats (n,2,3) maps output pixel centres -> input coords."""
    n, _, H, W = x.shape
    ys, xs = torch.meshgrid(torch.arange(H, device=x.device, dtype=x.dtype) + 0.5,
                            torch.arange(W, device=x.device, dtype=x.dtype) + 0.5, indexing='ij')
    pts = torch.stack([xs, ys, torch.ones_like(xs)], -1).view(1, H * W, 3)
    src = pts @ mats.transpose(1, 2)  # (n, HW, 2)
    grid = (src / torch.tensor([W, H], device=x.device, dtype=x.dtype) * 2 - 1).view(n, H, W, 2)
    out = F.grid_sample(x - FILL, grid, mode=mode, padding_mode='zeros', align_corners=False) + FILL
    return _q(out)


def _mats(n, device, a=1., b=0., c=0., d=0., e=1., f=0.):
    m = torch.zeros(n, 2, 3, device=device)
    for k, v in zip(range(6), (a, b, c, d, e, f)):
        m[:, k // 3, k % 3] = v
    return m


def op_shearX(x, m):
    return _affine(x, _mats(len(x), x.device, b=m), 'bicubic')


def op_shearY(x, m):
    return _affine(x, _mats(len(x), x.device, d=m), 'bicubic')


def op_translateX(x, m):
    return _affine(x, _mats(len(x), x.device, c=m * x.shape[-1]), 'nearest')


def op_translateY(x, m):
    return _affine(x, _mats(len(x), x.device, f=m * x.shape[-2]), 'nearest')


def op_rotate(x, m):
    H, W = x.shape[-2:]
    a = -m * math.pi / 180
    c, s = torch.cos(a), torch.sin(a)
    cx, cy = W / 2, H / 2
    mats = torch.stack([torch.stack([c, s, cx - c * cx - s * cy], -1),
                        torch.stack([-s, c, cy + s * cx - c * cy], -1)], 1)
    return _affine(x, mats, 'nearest')


def _gray(x):
    return (x[:, 0] * 0.299 + x[:, 1] * 0.587 + x[:, 2] * 0.114).round_().unsqueeze(1)


def _blend(degenerate, x, factor):
    return _q(degenerate + factor.view(-1, 1, 1, 1) * (x - degenerate))


def op_color(x, m):
    return _blend(_gray(x), x, 1 + m)


def op_contrast(x, m):
    mean = (_gray(x).mean(dim=(1, 2, 3), keepdim=True) + 0.5).floor()
    return _blend(mean, x, 1 + m)


def op_brightness(x, m):
    return _blend(torch.zeros_like(x), x, 1 + m)


def op_sharpness(x, m):
    k = torch.ones(3, 3, device=x.device, dtype=x.dtype)
    k[1, 1] = 5
    k = (k / 13).expand(3, 1, 3, 3)
    smooth = F.conv2d(x, k, groups=3).round_()
    deg = x.clone()
    deg[..., 1:-1, 1:-1] = smooth  # PIL keeps the border pixels
    return _blend(deg, x, 1 + m)


def op_posterize(x, m):
    div = (2.0 ** (8 - m)).view(-1, 1, 1, 1)
    return (x / div).floor_() * div


def op_solarize(x, m):
    t = m.view(-1, 1, 1, 1)
    return torch.where(x >= t, 255 - x, x)


def op_autocontrast(x, m):
    lo = x.amin(dim=(2, 3), keepdim=True)
    hi = x.amax(dim=(2, 3), keepdim=True)
    scale = 255.0 / (hi - lo).clamp_min(1)
    out = ((x - lo) * scale).floor_().clamp_(0, 255)
    return torch.where(hi > lo, out, x)


def op_equalize(x, m):
    n, C, H, W = x.shape
    xi = x.long().reshape(n * C, H * W)
    hist = torch.zeros(n * C, 256, device=x.device, dtype=torch.long).scatter_add_(1, xi, torch.ones_like(xi))
    nz = hist > 0
    last = 255 - nz.flip(1).long().argmax(1, keepdim=True)
    step = (H * W - hist.gather(1, last)) // 255  # (nC,1), as in PIL.ImageOps.equalize
    cum = hist.cumsum(1) - hist
    lut = ((step // 2 + cum) // step.clamp_min(1)).clamp_(max=255)
    identity = (step == 0) | (nz.sum(1, keepdim=True) <= 1)
    lut = torch.where(identity, torch.arange(256, device=x.device).expand_as(lut), lut)
    return lut.gather(1, xi).view(n, C, H, W).to(x.dtype)


def op_invert(x, m):
    return 255 - x


OPS = {name: globals()[f'op_{name}'] for name in OP_NAMES}


class GPUAutoAugment:
    """Batched CIFAR10Policy. x: float (B,3,H,W) with values in [0,255]."""

    def __init__(self, policy=CIFAR10_POLICY):
        P = len(policy)
        self.op = np.zeros((P, 2), np.int64)
        self.prob = np.zeros((P, 2))
        self.mag = np.zeros((P, 2))
        self.signed = np.zeros((P, 2), bool)
        for i, (p1, o1, m1, p2, o2, m2) in enumerate(policy):
            for s, (p, o, mi) in enumerate(((p1, o1, m1), (p2, o2, m2))):
                self.op[i, s] = OP_NAMES.index(o)
                self.prob[i, s] = p
                self.mag[i, s] = float(RANGES[o][mi])
                self.signed[i, s] = o in SIGNED

    def __call__(self, x, rng):
        B = x.shape[0]
        pol = rng.integers(0, len(self.op), B)
        for s in range(2):
            op = self.op[pol, s]
            apply = rng.random(B) < self.prob[pol, s]
            sign = np.where(self.signed[pol, s], rng.choice([-1.0, 1.0], B), 1.0)
            mag = self.mag[pol, s] * sign
            for t, name in enumerate(OP_NAMES):
                idx = np.nonzero(apply & (op == t))[0]
                if len(idx) == 0:
                    continue
                it = torch.from_numpy(idx).to(x.device, non_blocking=True)
                mt = torch.from_numpy(mag[idx].astype(np.float32)).to(x.device, non_blocking=True)
                x[it] = OPS[name](x[it], mt)
        return x


class GPULoader:
    """Iterates (images, targets) batches that are augmented on the GPU.

    Epoch semantics match finetune.py:
      * single GPU: RASampler(len, bs, 1, ra) -> `ra` fresh permutations per epoch
      * DDP: DistributedSampler + BatchAugmentation -> each index repeated `ra`
        times consecutively, sharded across ranks.
    """

    def __init__(self, x, y, batch_size, mean, std, device, train=True, aa=True, re=0.25, re_sh=0.4,
                 re_r1=0.3, re_sl=0.02, ra=3, distributed=False, rank=0, world_size=1, seed=0,
                 chunk=8192):
        self.x = torch.from_numpy(np.ascontiguousarray(x)).to(device).permute(0, 3, 1, 2).contiguous()  # uint8 NCHW
        self.y = torch.from_numpy(np.asarray(y, dtype=np.int64)).to(device)
        self.n = len(y)
        self.bs = batch_size
        self.device = device
        self.mean = torch.tensor(mean, device=device).view(1, 3, 1, 1)
        self.std = torch.tensor(std, device=device).view(1, 3, 1, 1)
        self.train, self.re, self.re_sh, self.re_r1, self.re_sl = train, re, re_sh, re_r1, re_sl
        self.ra = max(1, ra)
        self.distributed, self.rank, self.world = distributed, rank, world_size
        self.seed, self.epoch = seed, 0
        self.rng = np.random.default_rng(seed + 1000 * rank)
        self.autoaug = GPUAutoAugment() if (train and aa) else None
        self.chunk = max(batch_size, chunk // batch_size * batch_size)
        self.erase_fill = torch.tensor(mean, device=device).view(1, 3, 1, 1)  # same as RandomErasing(mean=stat[0])

    # ----------------------------------------------------------------- utils
    def set_epoch(self, epoch):
        self.epoch = epoch

    def _indices(self):
        if not self.train:
            return np.arange(self.n)
        if self.distributed:
            g = np.random.default_rng(self.seed + self.epoch)  # same permutation on every rank
            perm = g.permutation(self.n)
            total = int(math.ceil(self.n / self.world)) * self.world
            perm = np.concatenate([perm, perm[:total - self.n]])[self.rank::self.world]
            idx = np.repeat(perm, self.ra)
        else:
            idx = np.concatenate([self.rng.permutation(self.n) for _ in range(self.ra)])
        return idx[:len(idx) // self.bs * self.bs]  # drop_last

    def __len__(self):
        if not self.train:
            return int(math.ceil(self.n / self.bs))
        per = int(math.ceil(self.n / self.world)) if self.distributed else self.n
        return per * self.ra // self.bs

    def _crop_flip(self, x):
        B, C, H, W = x.shape
        flip = torch.from_numpy(self.rng.random(B) < 0.5).to(self.device, non_blocking=True)
        x = torch.where(flip.view(-1, 1, 1, 1), x.flip(3), x)
        xp = F.pad(x, (4, 4, 4, 4))
        oy = torch.from_numpy(self.rng.integers(0, 9, B)).to(self.device, non_blocking=True)
        ox = torch.from_numpy(self.rng.integers(0, 9, B)).to(self.device, non_blocking=True)
        ar_h = torch.arange(H, device=self.device)
        ar_w = torch.arange(W, device=self.device)
        rows = (oy[:, None] + ar_h)[:, :, None]
        cols = (ox[:, None] + ar_w)[:, None, :]
        b = torch.arange(B, device=self.device)[:, None, None]
        return xp[b, :, rows, cols].permute(0, 3, 1, 2)  # (B,H,W,C) -> (B,C,H,W)

    def _erase(self, x):
        B, C, H, W = x.shape
        K = 10  # attempts (reference uses up to 100; >99.9% succeed within 10)
        rng = self.rng
        area = H * W * rng.uniform(self.re_sl, self.re_sh, (B, K))
        ar = rng.uniform(self.re_r1, 1 / self.re_r1, (B, K))
        h = np.round(np.sqrt(area * ar)).astype(np.int64)
        w = np.round(np.sqrt(area / ar)).astype(np.int64)
        ok = (w < W) & (h < H)
        first = ok.argmax(1)
        any_ok = ok[np.arange(B), first] & (rng.random(B) < self.re)
        h = np.where(any_ok, h[np.arange(B), first], 0)
        w = np.where(any_ok, w[np.arange(B), first], 0)
        x1 = (rng.random(B) * (H - h + 1)).astype(np.int64)
        y1 = (rng.random(B) * (W - w + 1)).astype(np.int64)
        box = torch.from_numpy(np.stack([x1, x1 + h, y1, y1 + w], 1)).to(self.device, non_blocking=True)
        r = torch.arange(H, device=self.device).view(1, H, 1)
        c = torch.arange(W, device=self.device).view(1, 1, W)
        mask = ((r >= box[:, 0, None, None]) & (r < box[:, 1, None, None]) &
                (c >= box[:, 2, None, None]) & (c < box[:, 3, None, None])).unsqueeze(1)
        return torch.where(mask, self.erase_fill, x)

    def _transform(self, idx_t):
        x = self.x[idx_t].float()
        if self.train:
            x = self._crop_flip(x)
            if self.autoaug is not None:
                x = self.autoaug(x, self.rng)
        x = (x / 255.0 - self.mean) / self.std
        if self.train and self.re > 0:
            x = self._erase(x)
        return x

    def __iter__(self):
        idx = self._indices()
        step = self.chunk if self.train else self.bs
        for s in range(0, len(idx), step):
            idx_t = torch.from_numpy(idx[s:s + step]).to(self.device, non_blocking=True)
            xs, ys = self._transform(idx_t), self.y[idx_t]
            for b in range(0, len(idx_t), self.bs):
                yield xs[b:b + self.bs], ys[b:b + self.bs]
