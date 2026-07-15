# MTGF: Multi-Timestep Gradient Features for Membership Inference Attacks on Diffusion Models

Professional and reproducible implementation of Multi-Timestep Gradient Features (MTGF) for Membership Inference Attacks against Diffusion Models on the CIFAR-10 dataset.

The pipeline studies membership leakage from a DDPM target model using:

1. Multi-timestep reconstruction loss
2. 2D DCT residual decomposition
3. Low / mid / high frequency bands
4. Multi-Scale Temporal Gradient Features (MTGF)
5. Logistic Regression membership classifier

MTGF is the proposed feature extraction module. The engineering refactor does
not change the DDPM architecture, feature logic, MTGF computation, classifier,
or metrics.

## Repository Layout

```text
.
├── run_pipeline.py
├── config.py
├── requirements.txt
├── setup.py
├── training/
│   ├── train_target.py
│   └── train_shadow.py
├── feature_extraction/
│   └── extract_mtgf.py
├── evaluation/
│   ├── evaluate.py
│   └── metrics.py
├── visualization/
│   ├── plots.py
│   └── generate_figures.py
├── mia_code/
│   ├── model.py
│   ├── train.py
│   └── leakage_features.py
├── checkpoints/
├── data/
├── graphs/
├── logs/
└── results/
```

## Quick Start

```bash
git clone https://github.com/yashsinghrawat-ui/MTGF-Membership-Inference.git
cd MTGF-Membership-Inference
pip install -r requirements.txt
python run_pipeline.py
```

The pipeline will:

1. detect GPU availability,
2. check Python dependencies,
3. download CIFAR-10 if missing,
4. create output folders,
5. train or load the target DDPM,
6. train or load shadow DDPMs,
7. generate multi-timestep reconstruction features,
8. extract MTGF features,
9. train Logistic Regression,
10. evaluate and generate figures.

## Configuration

All primary parameters live in [config.py](config.py):

- dataset path
- random seed
- number of samples
- target/shadow epochs
- batch size
- learning rate
- diffusion timesteps
- classifier settings
- checkpoint/result/graph paths

## Outputs

Default artifacts:

```text
checkpoints/target_model.pth
checkpoints/shadows/shadow_model_*.pth
results/std_results.csv
results/std_results_labeled.csv
results/temporal_gradient_features.csv
results/research_results_v2.csv
graphs/Figure8_TemporalGradientComparison.png
```

## Method Summary

For each input image, the target DDPM predicts noise at multiple diffusion
timesteps. Reconstruction residuals are decomposed using 2D DCT into low, mid,
and high frequency bands. MTGF then computes differences of TSVA features
between adjacent timesteps:

```text
50→100, 100→150, 150→200, 200→300, 300→400, 400→500, 500→750
```

The resulting MTGF vector is used by Logistic Regression to predict whether an
image is a member or non-member of the target model's training set.

## Reproducibility Notes

- Seeds are centralized in `config.py`.
- Existing checkpoints and output CSVs are reused when available.
- Re-run from scratch by removing `checkpoints/`, `results/*.csv`, and `graphs/*.png`.
- The DDPM implementation remains in `mia_code/model.py`.
- Training remains in `mia_code/train.py`.
- MTGF logic is preserved from `temporal_gradient_experiment.py`.

## License

See [LICENSE](LICENSE).
