import os
import json
import torch
from PIL import Image
import torchvision.transforms as T

from mia_code.model import ImprovedUNet, make_schedule

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load config
with open("results/config.json", "r") as f:
    cfg = json.load(f)

# Load model
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

# Image path from terminal
import sys

if len(sys.argv) < 2:
    print("Usage: python predict_image.py image.jpg")
    exit()

img_path = sys.argv[1]

# Load image
img = Image.open(img_path).convert("RGB")

transform = T.Compose([
    T.Resize((32, 32)),
    T.ToTensor(),
    T.Normalize([0.5]*3, [0.5]*3)
])

x = transform(img).unsqueeze(0).to(device)

# Score calculation
with torch.no_grad():
    t_val = 100

    ab = alpha_bar[t_val]

    noise = torch.randn_like(x)

    xt = ab.sqrt() * x + (1 - ab).sqrt() * noise

    t_tensor = torch.tensor([t_val], device=device)

    pred = model(xt, t_tensor)

    score = ((pred - noise) ** 2).mean().item()

print("\n====================")
print("Image :", img_path)
print("Score :", score)
print("====================")