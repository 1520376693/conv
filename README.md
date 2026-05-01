# DAS Conv-TasNet Multi-Source Separation

This project implements a graduation-design experiment system for:

> Research on DAS multi-source signal separation based on Conv-TasNet

The system supports synthetic 2/3/4-source mixture generation from single-event DAS `.mat` samples, multi-channel Conv-TasNet separation, classification of separated signals, and qualitative/quantitative analysis.

## Structure

```text
src/das_sep/        Core package: data, preprocessing, models, losses, training, evaluation
scripts/            CLI entrypoints for training, evaluation, and smoke tests
configs/            Local smoke-test and RTX 4090 training configs
das_convtasnet_v2/  Original scripts kept for compatibility
```

Generated data, checkpoints, metrics, and figures are ignored by Git.

## Dataset

The expected dataset structure is:

```text
DAS-dataset/
  train/
    01_background/
    02_dig/
    03_knock/
    04_water/
    05_shake/
    06_walk/
  test/
    01_background/
    02_dig/
    03_knock/
    04_water/
    05_shake/
    06_walk/
```

Each sample is loaded as `[10000, 12]` and converted to `[12, T]`.

## Local Smoke Test

```powershell
py scripts/smoke_test.py --config configs/smoke_local.yml
```

This checks data loading, random mixing, separator forward pass, PIT loss backward pass, and classifier forward pass.

## RTX 4090 Training

On the cloud machine:

```bash
cd ~/conv
python scripts/train_classifier.py --config configs/train_4090.yml
python scripts/train_separator.py --config configs/train_4090.yml
python scripts/evaluate_pipeline.py --config configs/train_4090.yml \
  --separator_ckpt checkpoints/separator_4090/best.pt \
  --classifier_ckpt checkpoints/classifier_4090/best.pt \
  --out_dir results/metrics \
  --max_batches 300
```

Recommended order:

1. Train classifier on real single-event samples.
2. Train separator on 2/3/4-source random mixtures.
3. Evaluate separated-source classification and separation metrics.

## Metrics

Separation metrics:

- SI-SNR
- SI-SNRi
- SNR
- SDR
- MSE
- MAE
- Pearson correlation

Classification metrics:

- Accuracy
- Macro precision
- Macro recall
- Macro F1
- Confusion matrix

Target acceptance: separated-source classification accuracy >= 80%.
