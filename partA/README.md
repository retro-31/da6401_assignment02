# Part A: Training a CNN from Scratch

This directory implements training a small Convolutional Neural Network (CNN) from scratch on a subset of the iNaturalist dataset.

## Structure

```bash
partA/
├── data/               # iNaturalist subset (train, test)
├── models/             # Saved model checkpoints (best.ckpt)
├── scripts/            # Training, hyperparameter sweep, and prediction scripts
│   ├── train.py        # Train CNN from scratch and evaluate
│   ├── sweep_config.yaml # WandB Bayesian sweep configuration
│   └── show_predictions.py # Generate 10×3 grid of test predictions
├── utils/              # Utility modules
│   ├── model.py        # FlexibleCNN definition
│   └── data_loader.py  # get_dataloaders function
└── logs/               # WandB run logs
```

## Prerequisites

- Python 3.10 (via Conda)
- NVIDIA GPU with CUDA 11.8+ drivers
- Install dependencies:
  ```bash
  conda create -n da6401_partA python=3.10 -y
  conda activate da6401_partA
  pip install torch torchvision torchaudio pytorch-lightning wandb matplotlib
  ```

## Usage

### 1. Training the CNN

Train a CNN with configurable hyperparameters:
```bash
cd partA
python scripts/train.py \
  --num_filters 32 \
  --activation ReLU \
  --filter_organisation same \
  --data_augmentation True \
  --use_batchnorm True \
  --dropout_rate 0.2 \
  --batch_size 32 \
  --max_epochs 10 \
  --lr 1e-3
```
- **Best checkpoint** saved at `models/best.ckpt`.
- **WandB project**: `da6401-partA-sweep`.

### 2. Hyperparameter Sweep

Run a Bayesian hyperparameter sweep to tune: number of filters, activation, filter organization, augmentation, batch norm, dropout, batch size, and learning rate.
```bash
cd partA
wandb sweep scripts/sweep_config.yaml
wandb agent <entity>/da6401-partA-sweep/<sweep_id>
```

### 3. Generating Sample Predictions

After training or sweep completion, generate a creative 10×3 grid of test images and predictions:
```bash
cd partA
python scripts/show_predictions.py
```
- Uses `models/best.ckpt` and logs `sample_predictions_grid` to WandB.

## Outputs

- **Checkpoints**: `partA/models/best.ckpt`
- **Prediction grid**: logged as `sample_predictions_grid` in WandB.
- **Logs**: viewable under `da6401-partA-sweep` on WandB.

---

For detailed implementation, refer to each script’s header comments and docstrings.

