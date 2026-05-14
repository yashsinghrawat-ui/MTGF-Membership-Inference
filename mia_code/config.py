CFG = dict(
    # ------------------------------------------------------------------
    # Universe & splits
    # ------------------------------------------------------------------
    n_total       = 50000,   # entire CIFAR-10 train set is the universe
    target_frac   = 0.5,     # target trains on 25K (random 50% of universe)
    n_eval        = 2000,    # member probe samples
    n_nonmember   = 2000,    # non-member probe samples

    # ------------------------------------------------------------------
    n_shadow      = 16,      # 16 shadows => ~8 N_IN + ~8 N_OUT per probe sample
    shadow_frac   = 0.5,     # each shadow trains on 25K of the same universe

    # ------------------------------------------------------------------
    # Training hyper-params  
    # ------------------------------------------------------------------
    target_epochs = 800,     
    shadow_epochs = 300,    
    batch_size    = 128,     
    lr            = 2e-4,    
    weight_decay  = 1e-4,    
    grad_clip     = 1.0,
    print_every   = 20,      

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
