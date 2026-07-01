import hashlib
import math
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence

import torch


TIMESTEPS: List[int] = [50, 100, 150, 200, 300, 400, 500, 750]
BANDS: List[str] = ["low", "mid", "high"]
STATS: List[str] = ["mean", "std"]
RAW_LOSS_TIMESTEP = 100
DEFAULT_N_MC = 50

EARLY_TIMESTEPS: List[int] = [50, 100, 150, 200]
MID_TIMESTEPS: List[int] = [300, 400]
LATE_TIMESTEPS: List[int] = [400, 500, 750]


def stable_seed(value: str, base_seed: int = 12345) -> int:
    digest = hashlib.sha256(f"{base_seed}:{value}".encode("utf-8")).hexdigest()
    return int(digest[:8], 16)


def feature_names(
    timesteps: Sequence[int] = TIMESTEPS,
    bands: Sequence[str] = BANDS,
    stats: Sequence[str] = STATS,
) -> List[str]:
    names = ["raw_loss"]

    for t in timesteps:
        names.extend([f"raw_loss_mean_{t}", f"raw_loss_std_{t}"])
        for stat in stats:
            for band in bands:
                names.append(f"{band}_{stat}_{t}")

    return names


def tsva_v2_columns(
    timesteps: Sequence[int] = TIMESTEPS,
    bands: Sequence[str] = BANDS,
    stats: Sequence[str] = STATS,
) -> List[str]:
    return [
        f"{band}_{stat}_{t}"
        for t in timesteps
        for stat in stats
        for band in bands
    ]


def raw_loss_columns(timesteps: Sequence[int] = TIMESTEPS) -> List[str]:
    cols = ["raw_loss"]
    for t in timesteps:
        cols.extend([f"raw_loss_mean_{t}", f"raw_loss_std_{t}"])
    return cols


def timestep_columns(timestep: int) -> List[str]:
    return [
        f"raw_loss_mean_{timestep}",
        f"raw_loss_std_{timestep}",
        *[
            f"{band}_{stat}_{timestep}"
            for stat in STATS
            for band in BANDS
        ],
    ]


def tsva_columns_for_timesteps(timesteps: Iterable[int]) -> List[str]:
    ts = set(timesteps)
    return [c for c in tsva_v2_columns() if int(c.rsplit("_", 1)[1]) in ts]


def raw_columns_for_timesteps(
    timesteps: Iterable[int],
    include_canonical: bool = False,
) -> List[str]:
    ts = set(timesteps)
    cols = []
    if include_canonical:
        cols.append("raw_loss")
    for t in TIMESTEPS:
        if t in ts:
            cols.extend([f"raw_loss_mean_{t}", f"raw_loss_std_{t}"])
    return cols


def tsva_columns_for_bands(bands: Iterable[str]) -> List[str]:
    selected = set(bands)
    return [
        c for c in tsva_v2_columns()
        if c.split("_", 1)[0] in selected
    ]


def tsva_columns_for_stats(stats: Iterable[str]) -> List[str]:
    selected = set(stats)
    return [
        c for c in tsva_v2_columns()
        if c.split("_", 2)[1] in selected
    ]


def dct2(x: torch.Tensor) -> torch.Tensor:
    squeezed = x.dim() == 3
    if squeezed:
        x = x.unsqueeze(0)

    _, _, h, w = x.shape
    xm = torch.cat([x, x.flip(-2)], 2)
    xm = torch.cat([xm, xm.flip(-1)], 3)
    values = torch.fft.rfft2(xm, norm="ortho")[:, :, :h, : w // 2 + 1]

    kh = torch.arange(h, device=x.device).float() / (2 * h)
    kw = torch.arange(w // 2 + 1, device=x.device).float() / (2 * w)
    ph = torch.exp(-1j * math.pi * kh).reshape(1, 1, h, 1)
    pw = torch.exp(-1j * math.pi * kw).reshape(1, 1, 1, -1)

    out = (values * ph * pw).real
    return out.squeeze(0) if squeezed else out


def band_mask(height: int, width: int, band: str, device: torch.device) -> torch.Tensor:
    cutoffs = {
        "low": (0, 4),
        "mid": (4, 12),
        "high": (12, 9999),
    }
    if band not in cutoffs:
        raise ValueError(f"Unknown spectral band: {band}")

    r1, r2 = cutoffs[band]
    i = torch.arange(height, device=device).float()
    j = torch.arange(width // 2 + 1, device=device).float()
    radius = torch.sqrt(i[:, None] ** 2 + j[None, :] ** 2)
    return ((radius >= r1) & (radius < r2)).float()


@dataclass
class LeakageFeatureExtractor:
    model: torch.nn.Module
    alpha_bar: torch.Tensor
    device: torch.device
    n_mc: int = DEFAULT_N_MC
    base_seed: int = 12345

    def __post_init__(self) -> None:
        self.model.eval()
        self.masks = {
            band: band_mask(32, 32, band, self.device)
            for band in BANDS
        }

    def _seed_noise(self, image_id: str, timestep: int, repeat: int) -> None:
        seed = stable_seed(f"{image_id}:t={timestep}:mc={repeat}", self.base_seed)
        torch.manual_seed(seed)
        if self.device.type == "cuda":
            torch.cuda.manual_seed_all(seed)

    @torch.no_grad()
    def extract(self, x0: torch.Tensor, image_id: str) -> Dict[str, float]:
        row: Dict[str, float] = {}

        for timestep in TIMESTEPS:
            row.update(self._extract_timestep(x0, image_id, timestep))

        row["raw_loss"] = row[f"raw_loss_mean_{RAW_LOSS_TIMESTEP}"]
        return {name: row[name] for name in feature_names()}

    @torch.no_grad()
    def _extract_timestep(
        self,
        x0: torch.Tensor,
        image_id: str,
        timestep: int,
    ) -> Dict[str, float]:
        ab = self.alpha_bar[timestep]
        t_tensor = torch.full(
            (x0.shape[0],),
            timestep,
            device=self.device,
            dtype=torch.long,
        )

        raw_losses = []
        band_losses = {band: [] for band in BANDS}

        for repeat in range(self.n_mc):
            self._seed_noise(image_id, timestep, repeat)
            noise = torch.randn_like(x0)
            xt = ab.sqrt() * x0 + (1 - ab).sqrt() * noise
            pred = self.model(xt, t_tensor)

            err = pred - noise
            raw_losses.append(err.pow(2).mean(dim=[1, 2, 3]).detach().cpu())

            dct_err = dct2(err)
            for band, mask in self.masks.items():
                band_e2 = dct_err.pow(2) * mask[None, None]
                band_losses[band].append(
                    band_e2.mean(dim=[1, 2, 3]).detach().cpu()
                )

        row: Dict[str, float] = {}
        raw = torch.stack(raw_losses)
        row[f"raw_loss_mean_{timestep}"] = float(raw.mean(0).numpy()[0])
        row[f"raw_loss_std_{timestep}"] = float(raw.std(0).numpy()[0])

        for stat in STATS:
            for band in BANDS:
                values = torch.stack(band_losses[band])
                if stat == "mean":
                    row[f"{band}_{stat}_{timestep}"] = float(
                        values.mean(0).numpy()[0]
                    )
                elif stat == "std":
                    row[f"{band}_{stat}_{timestep}"] = float(
                        values.std(0).numpy()[0]
                    )
                else:
                    raise ValueError(f"Unsupported statistic: {stat}")

        return row
