import csv
import os
import sys
import json
import math
import numpy as np
import torch
import torch.fft
from PIL import Image
import torchvision.transforms as T

from mia_code.model import ImprovedUNet, make_schedule

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --------------------------------------------------
# Load Config
# --------------------------------------------------

with open("results/config.json", "r") as f:
    cfg = json.load(f)

# --------------------------------------------------
# Load Model
# --------------------------------------------------

print("Loading model...")

model = ImprovedUNet().to(device)

state = torch.load(
    "results/target_model.pth",
    map_location=device
)

model.load_state_dict(state)
model.eval()

alpha_bar = make_schedule(
    cfg["T"],
    cfg["beta_start"],
    cfg["beta_end"],
    device=device
)

# --------------------------------------------------
# SMS Parameters
# --------------------------------------------------

t_lira = 100

weights = np.array([
    -0.00132887,
    -0.00191238,
    -0.00108737
])

intercept = -0.02169458

# --------------------------------------------------
# DCT Functions
# --------------------------------------------------

def dct2(x):

    sq = x.dim() == 3

    if sq:
        x = x.unsqueeze(0)

    B, C, H, W = x.shape

    xm = torch.cat([x, x.flip(-2)], 2)
    xm = torch.cat([xm, xm.flip(-1)], 3)

    V = torch.fft.rfft2(xm, norm="ortho")[:, :, :H, :W//2 + 1]

    kh = torch.arange(H, device=x.device).float() / (2 * H)
    kw = torch.arange(W//2 + 1, device=x.device).float() / (2 * W)

    ph = torch.exp(-1j * math.pi * kh).reshape(1, 1, H, 1)
    pw = torch.exp(-1j * math.pi * kw).reshape(1, 1, 1, -1)

    out = (V * ph * pw).real

    return out.squeeze(0) if sq else out


def band_mask(H, W, band, dev="cpu"):

    cutoffs = {
        "low": (0, 4),
        "mid": (4, 12),
        "high": (12, 9999)
    }

    r1, r2 = cutoffs[band]

    I = torch.arange(H, device=dev).float()
    J = torch.arange(W//2 + 1, device=dev).float()

    R = torch.sqrt(I[:, None]**2 + J[None, :]**2)

    return ((R >= r1) & (R < r2)).float()


MASKS = {
    b: band_mask(32, 32, b, dev=device.type)
    for b in ("low", "mid", "high")
}

# --------------------------------------------------
# SMS Feature Extraction
# --------------------------------------------------

@torch.no_grad()
def spectral_band_losses(mdl, x0, t_val, n_mc=10):

    mdl.eval()

    ab = alpha_bar[t_val]

    acc = {
        "low": [],
        "mid": [],
        "high": []
    }

    tv = torch.full(
        (x0.shape[0],),
        t_val,
        device=device,
        dtype=torch.long
    )

    for _ in range(n_mc):

        eps = torch.randn_like(x0)

        xt = ab.sqrt() * x0 + (1 - ab).sqrt() * eps

        pred = mdl(xt, tv)

        err = dct2(pred - eps)

        for b, m in MASKS.items():

            e2 = (err ** 2) * m[None, None].to(device)

            acc[b].append(
                e2.mean([1, 2, 3]).cpu()
            )

    results = {}

    for b in acc:

        vals = torch.stack(acc[b])

        results[b] = {
            "mean": vals.mean(0).numpy(),
            "std": vals.std(0).numpy()
        }

    return results

# --------------------------------------------------
# Image Input
# --------------------------------------------------

if len(sys.argv) < 2:
    print("Usage:")
    print("python predict_image.py image.jpg")
    sys.exit()

img_path = sys.argv[1]

img = Image.open(img_path).convert("RGB")

transform = T.Compose([
    T.Resize((32, 32)),
    T.ToTensor(),
    T.Normalize([0.5]*3, [0.5]*3)
])

x = transform(img).unsqueeze(0).to(device)

# --------------------------------------------------
# Raw Diffusion Loss
# --------------------------------------------------

with torch.no_grad():

    ab = alpha_bar[t_lira]

    noise = torch.randn_like(x)

    xt = ab.sqrt() * x + (1 - ab).sqrt() * noise

    t_tensor = torch.tensor(
        [t_lira],
        device=device
    )

    pred = model(xt, t_tensor)

    raw_loss = (
        (pred - noise) ** 2
    ).mean().item()

# --------------------------------------------------
# SMS Features
# --------------------------------------------------

bands = spectral_band_losses(
    model,
    x,
    t_lira,
    n_mc=50
)

low = float(bands["low"]["mean"][0])
mid = float(bands["mid"]["mean"][0])
high = float(bands["high"]["mean"][0])

low_std = float(bands["low"]["std"][0])
mid_std = float(bands["mid"]["std"][0])
high_std = float(bands["high"]["std"][0])

# --------------------------------------------------
# SMS Score
# --------------------------------------------------

z = (
    low * weights[0]
    + mid * weights[1]
    + high * weights[2]
    + intercept
)

sms_score = 1.0 / (1.0 + np.exp(-z))

prediction = (
    "Likely MEMBER"
    if sms_score > 0.5
    else "Likely NON-MEMBER"
)
# --------------------------------------------------
# Save Results to CSV
# --------------------------------------------------

csv_file = "std_results.csv"

file_exists = os.path.isfile(csv_file)

with open(csv_file, "a", newline="") as f:

    writer = csv.writer(f)

    if not file_exists:
        writer.writerow([
            "image",
            "low_mean",
            "mid_mean",
            "high_mean",
            "low_std",
            "mid_std",
            "high_std",
            "sms_score",
            "prediction"
        ])

    writer.writerow([
        img_path,
        low,
        mid,
        high,
        low_std,
        mid_std,
        high_std,
        sms_score,
        prediction
    ])
# --------------------------------------------------
# --------------------------------------------------
# Output
# --------------------------------------------------

print("\n===================================")
print("Image           :", img_path)

print("Raw Loss        :", raw_loss)

print("Low Band Mean   :", low)
print("Mid Band Mean   :", mid)
print("High Band Mean  :", high)

print("Low Band Std    :", low_std)
print("Mid Band Std    :", mid_std)
print("High Band Std   :", high_std)

print("SMS Score       :", sms_score)
print("Prediction      :", prediction)

print("===================================\n")