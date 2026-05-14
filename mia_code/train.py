import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .model import ImprovedUNet, make_schedule


def train_model(dataset, epochs: int, cfg: dict, device: torch.device,
                model_name: str = "model"):
   
    T         = cfg["T"]
    alpha_bar = make_schedule(T, cfg["beta_start"], cfg["beta_end"], device=device)

    model     = ImprovedUNet().to(device)

    # AdamW with weight decay for better regularisation
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr           = cfg["lr"],
        weight_decay = cfg.get("weight_decay", 1e-4),
    )

    # Cosine annealing: gently decays LR to lr_min over training
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max   = epochs,
        eta_min = cfg["lr"] * 0.1,   # decay to 10% of initial LR
    )

    use_amp = (device.type == "cuda")
    scaler  = torch.cuda.amp.GradScaler(enabled=use_amp)

    loader = DataLoader(
        dataset,
        batch_size        = cfg["batch_size"],
        shuffle           = True,
        num_workers       = 4,          # more workers to keep GPU fed
        pin_memory        = True,
        drop_last         = True,
        persistent_workers= True,       # avoids worker restart overhead
    )

    print_every = cfg.get("print_every", 20)   # log every N epochs

    history = []
    for epoch in range(epochs):
        model.train()
        total_loss, n = 0.0, 0

        for x, _ in loader:
            x  = x.to(device, non_blocking=True)
            bs = x.size(0)
            t  = torch.randint(0, T, (bs,), device=device)

            noise = torch.randn_like(x)
            ab    = alpha_bar[t].view(-1, 1, 1, 1)
            xt    = ab.sqrt() * x + (1 - ab).sqrt() * noise

            # ── AMP forward ──────────────────────────────────────
            with torch.cuda.amp.autocast(enabled=use_amp):
                pred = model(xt, t)
                loss = F.mse_loss(pred, noise)

            # ── AMP backward + grad-clip + step ──────────────────
            optimizer.zero_grad(set_to_none=True)   # faster than zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item() * bs
            n          += bs

        scheduler.step()
        epoch_loss = total_loss / n
        history.append(epoch_loss)

        if (epoch + 1) % print_every == 0 or epoch == 0 or (epoch + 1) == epochs:
            cur_lr = scheduler.get_last_lr()[0]
            print(
                f"[{model_name}] epoch {epoch+1:>5}/{epochs}"
                f" | loss={epoch_loss:.5f}"
                f" | lr={cur_lr:.2e}",
                flush=True,
            )

    return model, history
