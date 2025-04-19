# Part B: Fine-Tuning a Pre-trained Model

This directory implements fine-tuning a ResNet50 model (pre-trained on ImageNet) on a subset of the iNaturalist dataset.

## Structure

```bash
partB/
├── data/               # iNaturalist subset (train, test)
├── models/             # Saved model checkpoints (best.ckpt)
├── scripts/            # Fine-tuning, hyperparameter sweep, and prediction scripts
│   ├── train_finetune.py   # Fine-tune and evaluate model
│   ├── sweep_config.yaml   # WandB Bayesian sweep configuration
│   └── show_predictions.py # Generate 10×3 grid of test predictions
├── utils/              # Utility modules
│   └── data_loader.py  # get_dataloaders function
└── logs/               # WandB run logs
```

## Prerequisites

- Python 3.10 (via Conda environment)
- NVIDIA GPU with CUDA drivers
- Install dependencies:
  ```bash
  conda create -n da6401_partB python=3.10 -y
  conda activate da6401_partB
  pip install torch torchvision torchaudio pytorch-lightning wandb matplotlib
  ```

## Usage

### 1. Fine-Tuning the Pre-trained Model

```bash
cd partB
python scripts/train_finetune.py \
  --unfreeze_layers 2 \
  --batch_size 32 \
  --max_epochs 10 \
  --lr 1e-4
```

- Saves the best checkpoint to `models/best.ckpt`.
- Logs training/validation metrics to the WandB project `da6401-partB-sweep`.

### 2. Hyperparameter Sweep

```bash
cd partB
wandb sweep scripts/sweep_config.yaml
wandb agent <entity>/da6401-partB-sweep/<sweep_id>
```

- Sweeps over `unfreeze_layers`, `batch_size`, `lr`, and `max_epochs` to optimize validation accuracy.

### 3. Generating Sample Predictions

```bash
cd partB
python scripts/show_predictions.py
```

- Logs a 10×3 grid of test images with predicted labels to WandB under the key `sample_predictions`.

## Outputs

- **Checkpoint**: `partB/models/best.ckpt`
- **Prediction grid**: logged in WandB (Media).

Refer to individual scripts for detailed implementation notes and further customization options.

