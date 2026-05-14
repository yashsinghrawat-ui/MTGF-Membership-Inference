import modal
import torch
import random
import time
import os
import json
import numpy as np

from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import torch.nn.functional as F

from mia_code.config import CFG
from mia_code.model  import make_schedule
from mia_code.train  import train_model

# ─────────────────────────────────────────────────────────────────
#  Modal infrastructure
# ─────────────────────────────────────────────────────────────────

app = modal.App("mia-diffusion")

image = (
    modal.Image.debian_slim()
    .pip_install_from_requirements("requirements.txt")
    .add_local_python_source("mia_code")
)

volume  = modal.Volume.from_name("mia-diffusion-results", create_if_missing=True)
OUTPUTS = "/outputs"


# ─────────────────────────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────────────────────────

def log(msg: str):
    ts = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())
    print(f"[{ts} UTC] {msg}", flush=True)


def save_json(obj, path: str):
    with open(path, "w") as f:
        json.dump(obj, f)
    log(f"[SAVE] {path}  ({os.path.getsize(path) / 1024:.1f} KB)")


@torch.no_grad()
def eval_losses(model: torch.nn.Module,
                x_cpu: torch.Tensor,
                probe_ts: list,
                alpha_bar: torch.Tensor,
                device: torch.device,
                batch_size: int = 256) -> np.ndarray:
    """
    Evaluate model loss at each probe timestep for all samples in x_cpu.
    Uses autocast (AMP) for faster forward passes during evaluation.

    Returns
    -------
    losses : float32 ndarray of shape (len(probe_ts), N)
    """
    model.eval()
    use_amp = (device.type == "cuda")
    N       = x_cpu.shape[0]
    result  = np.zeros((len(probe_ts), N), dtype=np.float32)

    for ti, t_val in enumerate(probe_ts):
        t_val = min(t_val, alpha_bar.shape[0] - 1)
        col   = []
        for start in range(0, N, batch_size):
            xb       = x_cpu[start : start + batch_size].to(device, non_blocking=True)
            bs       = xb.size(0)
            t_tensor = torch.full((bs,), t_val, device=device, dtype=torch.long)
            noise    = torch.randn_like(xb)
            ab       = alpha_bar[t_val]
            xt       = ab.sqrt() * xb + (1 - ab).sqrt() * noise
            with torch.cuda.amp.autocast(enabled=use_amp):
                pred = model(xt, t_tensor)
            loss_per = ((pred.float() - noise) ** 2).mean(dim=[1, 2, 3])
            col.append(loss_per.cpu().numpy())
        result[ti] = np.concatenate(col)
        log(f"      t={t_val:4d} | mean_loss={result[ti].mean():.4f}")

    return result


def _load_subset_cpu(dataset, indices: list) -> torch.Tensor:
    """Load a list of dataset indices into a CPU float tensor."""
    loader = DataLoader(
        Subset(dataset, indices),
        batch_size=512, shuffle=False, num_workers=4, pin_memory=False,
    )
    return torch.cat([xb for xb, _ in loader], dim=0)


# ─────────────────────────────────────────────────────────────────
#  Pipeline class
# ─────────────────────────────────────────────────────────────────

@app.cls(
    image=image,
    gpu="H100",
    volumes={OUTPUTS: volume},
    timeout=86400,          # 24-hour max (job takes ~3-5 h in practice)
)
class Pipeline:

    @modal.enter()
    def setup(self):
        log("═" * 60)
        log("  SETUP — MIA Diffusion v2 (correct LiRA design)")
        log("═" * 60)

        self.device = torch.device("cuda")
        log(f"GPU  : {torch.cuda.get_device_name(0)}")
        log(f"VRAM : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

        # CIFAR-10 (no augmentation — we want clean, deterministic losses)
        tfm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5] * 3, [0.5] * 3),
        ])
        log("Downloading CIFAR-10 …")
        self.full_ds = datasets.CIFAR10(root="/data", train=True,
                                        download=True, transform=tfm)
        log(f"CIFAR-10 ready | {len(self.full_ds):,} samples")

        # ── Noise schedule ──────────────────────────────────────
        self.alpha_bar = make_schedule(CFG["T"], CFG["beta_start"],
                                       CFG["beta_end"], device=self.device)
        self.probe_ts  = CFG["t_sweep"]

    # ──────────────────────────────────────────────────────────────
    #  Stage 1: Train target model
    # ──────────────────────────────────────────────────────────────
    def _train_target(self):
        log("═" * 60)
        log("  STAGE 1 — TARGET MODEL")
        log("═" * 60)

        n_total  = CFG["n_total"]           # 50K
        n_target = int(n_total * CFG["target_frac"])  # 25K

        # ── Build universe split ────────────────────────────────
        idx = list(range(n_total))
        random.shuffle(idx)

        target_idx_list  = idx[:n_target]           # 25K members
        nontarget_list   = idx[n_target:]           # 25K non-members

        self.target_idx_set  = set(target_idx_list)
        self.target_idx_list = target_idx_list
        self.nontarget_list  = nontarget_list

        # ── Challenge set ───────────────────────────────────────
        # Members:     first n_eval from target's 25K
        # Non-members: first n_nonmember from the other 25K
        n_eval      = CFG["n_eval"]
        n_nonmember = CFG["n_nonmember"]

        self.member_eval_idx    = target_idx_list[:n_eval]
        self.nonmember_eval_idx = nontarget_list[:n_nonmember]
        self.challenge_idx      = self.member_eval_idx + self.nonmember_eval_idx
        # Layout: [0 : n_eval] = members  |  [n_eval : n_eval+n_nonmember] = non-members

        log(f"Universe    : {n_total:,} samples")
        log(f"Target set  : {n_target:,} members")
        log(f"Non-target  : {len(nontarget_list):,} samples")
        log(f"Challenge   : {len(self.challenge_idx):,} total "
            f"({n_eval} members + {n_nonmember} non-members)")

        # ── Save metadata ───────────────────────────────────────
        os.makedirs(OUTPUTS, exist_ok=True)
        save_json(CFG,                      f"{OUTPUTS}/config.json")
        save_json(target_idx_list,          f"{OUTPUTS}/target_idx.json")
        save_json(self.member_eval_idx,     f"{OUTPUTS}/target_sample_idx.json")
        save_json(self.nonmember_eval_idx,  f"{OUTPUTS}/nonmember_idx.json")
        volume.commit()
        log("Metadata committed.")

        # ── Pre-load challenge images (CPU) ─────────────────────
        log("Pre-loading challenge images …")
        self.challenge_x = _load_subset_cpu(self.full_ds, self.challenge_idx)
        log(f"  challenge_x : {self.challenge_x.shape}  "
            f"({self.challenge_x.nbytes / 1e6:.0f} MB)")

        # ── Train ───────────────────────────────────────────────
        log(f"Training target model for {CFG['target_epochs']} epochs on "
            f"{n_target:,} samples …")
        t0 = time.time()
        ds = Subset(self.full_ds, target_idx_list)
        m, _ = train_model(ds, CFG["target_epochs"], CFG, self.device,
                           model_name="target")
        log(f"Target training done in {(time.time() - t0) / 60:.1f} min")

        torch.save(m.state_dict(), f"{OUTPUTS}/target_model.pth")
        volume.commit()
        log("Target model committed.")

        del m
        torch.cuda.empty_cache()

    # ──────────────────────────────────────────────────────────────
    #  Stage 2: Train shadow models + pre-compute losses
    # ──────────────────────────────────────────────────────────────
    def _train_shadows(self):
        log("═" * 60)
        log("  STAGE 2 — SHADOW MODELS")
        log("═" * 60)

        n_shadow    = CFG["n_shadow"]
        n_total     = CFG["n_total"]
        n_split     = int(n_total * CFG["shadow_frac"])  # 25K per shadow
        probe_ts    = self.probe_ts
        n_challenge = len(self.challenge_idx)            # 4K

        # Pre-allocate arrays (crash-safe: flushed after every shadow)
        shadow_losses     = np.zeros((n_shadow, len(probe_ts), n_challenge),
                                     dtype=np.float32)
        shadow_membership = np.zeros((n_shadow, n_challenge), dtype=bool)

        t0 = time.time()

        for i in range(n_shadow):
            log(f"─── Shadow {i+1} / {n_shadow} ───")

            # ── Each shadow reproducibly picks 25K from ALL 50K ──────
            # Seed = global seed + shadow index so the split is identical
            # if modal_train_resume.py ever needs to reproduce this shadow.
            rng       = random.Random(CFG.get("seed", 42) + i * 1000)
            in_global = sorted(rng.sample(range(n_total), n_split))
            in_set    = set(in_global)

            # ── Track membership for challenge samples ────────────
            for j, ci in enumerate(self.challenge_idx):
                shadow_membership[i, j] = (ci in in_set)

            n_in  = shadow_membership[i].sum()
            n_out = n_challenge - n_in
            log(f"  shadow trained on {n_split:,} samples | "
                f"challenge: {n_in} in-shadow, {n_out} out-shadow")

            # ── Train ─────────────────────────────────────────────
            ds   = Subset(self.full_ds, in_global)
            m, _ = train_model(ds, CFG["shadow_epochs"], CFG, self.device,
                               model_name=f"shadow-{i+1}")

            # ── Save weights ──────────────────────────────────────
            torch.save(m.state_dict(), f"{OUTPUTS}/shadow_model_{i+1}.pth")

            # ── Evaluate on full challenge set ────────────────────
            log(f"  Evaluating shadow {i+1} on {n_challenge} challenge samples …")
            shadow_losses[i] = eval_losses(m, self.challenge_x, probe_ts,
                                           self.alpha_bar, self.device)

            # ── Flush to volume (crash protection) ────────────────
            np.save(f"{OUTPUTS}/shadow_losses.npy",     shadow_losses)
            np.save(f"{OUTPUTS}/shadow_membership.npy", shadow_membership)
            volume.commit()
            log(f"[SAVE] shadow {i+1} committed ✓\n")

            del m
            torch.cuda.empty_cache()

        elapsed = time.time() - t0
        log(f"All {n_shadow} shadow models done in {elapsed / 60:.1f} min "
            f"({elapsed / 3600:.2f} h)")

        # ── Final verification ────────────────────────────────────
        log(f"\nFinal array shapes:")
        log(f"  shadow_losses     : {shadow_losses.shape}")
        log(f"  shadow_membership : {shadow_membership.shape}")
        log(f"  in/out balance    : {shadow_membership.mean():.3f} (expect ~0.50)")

    # ──────────────────────────────────────────────────────────────
    #  Entry point
    # ──────────────────────────────────────────────────────────────
    @modal.method()
    def run(self):
        log("╔" + "═" * 58 + "╗")
        log("║   MIA Diffusion Training — v2 (Correct LiRA Design)    ║")
        log("╚" + "═" * 58 + "╝")
        log(f"Config: {CFG}")

        self._train_target()
        self._train_shadows()

        log("╔" + "═" * 58 + "╗")
        log("║   ALL DONE — Download with download_from_modal.py       ║")
        log("╚" + "═" * 58 + "╝")
        log("Files saved:")
        for fname in sorted(os.listdir(OUTPUTS)):
            fpath = os.path.join(OUTPUTS, fname)
            if os.path.isfile(fpath):
                log(f"  {fname:<45s} {os.path.getsize(fpath)/1e6:7.1f} MB")


# ─────────────────────────────────────────────────────────────────
#  Entry point
# ─────────────────────────────────────────────────────────────────

@app.local_entrypoint()
def main():
    Pipeline().run.remote()
