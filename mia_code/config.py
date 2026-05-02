"""
config.py -- Central configuration for the MIA diffusion pipeline.

v2 LiRA experimental design (correct setup):
  - Universe  = ALL 50K CIFAR-10 samples
  - Target    trains on random 25K (50% of universe)
  - Each shadow also trains on random 25K of the SAME 50K universe
  - This gives proper N_IN and N_OUT for EVERY challenge sample

H100 budget analysis (@ $0.001097 / sec):
  $30 limit = ~27,300 seconds of GPU time

  Without AMP (old code, batch=64):
    - epoch time ~4.9 sec  (80 steps/sec, 390 steps/epoch)
    - target  1000 epochs = 4,900 sec  = $5.4
    - 1 shadow 500 epochs = 2,450 sec  = $2.7  x16 = $43.2
    - TOTAL  ~$48.6  -- WAY OVER BUDGET

  With AMP + batch=128 (this config):
    - epoch time ~2.0 sec  (195 steps/epoch, 2x faster via FP16)
    - target  800 epochs = 1,600 sec  = $1.8
    - 1 shadow 400 epochs = 800 sec   = $0.9  x16 = $14.0
    - eval overhead per shadow           ~4 sec x16 = $0.07
    - TOTAL  ~$16  -- UNDER $30, with quality very close to paper

  Headroom: ~$14 safety margin covers Modal cold-start, data-loading, etc.
"""

CFG = dict(
    # ------------------------------------------------------------------
    # Universe & splits
    # ------------------------------------------------------------------
    n_total       = 50000,   # entire CIFAR-10 train set is the universe
    target_frac   = 0.5,     # target trains on 25K (random 50% of universe)
    n_eval        = 2000,    # member probe samples
    n_nonmember   = 2000,    # non-member probe samples

    # ------------------------------------------------------------------
    # Shadow models  (paper uses 16; we keep 16 because AMP gives budget)
    # ------------------------------------------------------------------
    n_shadow      = 16,      # 16 shadows => ~8 N_IN + ~8 N_OUT per probe sample
    shadow_frac   = 0.5,     # each shadow trains on 25K of the same universe

    # ------------------------------------------------------------------
    # Training hyper-params  (AMP-tuned)
    # ------------------------------------------------------------------
    target_epochs = 800,     # good memorisation; ~27 min on H100 with AMP
    shadow_epochs = 300,     # stable convergence; ~13 min each x16 = ~3.6 h
    batch_size    = 128,     # double vs baseline: better GPU utilisation
    lr            = 2e-4,    # linear-scale rule: 2x batch => 2x LR
    weight_decay  = 1e-4,    # AdamW regularisation (new field)
    grad_clip     = 1.0,
    print_every   = 20,      # log every N epochs (reduces Modal I/O overhead)

    # ------------------------------------------------------------------
    # DDPM schedule
    # ------------------------------------------------------------------
    T          = 1000,
    beta_start = 1e-4,
    beta_end   = 0.02,

    # ------------------------------------------------------------------
    # Evaluation timesteps  (used by modal_train.py + eval_kaggle.ipynb)
    # ------------------------------------------------------------------
    t_sweep  = [1, 5, 10, 50, 75, 100, 150, 200, 300, 400, 500, 750, 999],
    t_lira   = 100,                      # Goldilocks timestep for LiRA
    t_strong = [50, 100, 150, 200],      # timesteps for Strong LiRA sum
    n_repeats = 50,                      # noise samples per query (Strong LiRA)

    # ------------------------------------------------------------------
    # Kaggle / Colab evaluation
    # ------------------------------------------------------------------
    n_probe  = 2000,         # must equal n_eval = n_nonmember
)
