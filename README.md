# DA6401 Assignment 2: CNN Training & Fine-Tuning

This repository contains implementations for two key tasks of DA6401 Assignment 2:

1. **Part A**: Building and training a Convolutional Neural Network (CNN) from scratch.
2. **Part B**: Fine-tuning a pre-trained ResNet50 model.

Both parts utilize PyTorch Lightning for simplified training routines and WandB (Weights & Biases) for experiment tracking and hyperparameter tuning.

## Repository Structure

```yaml
da6401_assignment02/
├── partA/                   # CNN from scratch
│   ├── data/                # Dataset (train, test)
│   ├── models/              # Model checkpoints
│   ├── scripts/             # Training & prediction scripts
│   ├── utils/               # Utility scripts
│   └── README.md            # Part A instructions
│
├── partB/                   # Fine-tuning ResNet50
│   ├── data/                # Dataset (train, test)
│   ├── models/              # Model checkpoints
│   ├── scripts/             # Fine-tuning & prediction scripts
│   ├── utils/               # Utility scripts
│   └── README.md            # Part B instructions
│
└── README.md                # This file
```

## Environment Setup

### Step 1: Create and activate a Conda environment

```bash
conda create -n da6401 python=3.10 -y
conda activate da6401
```

### Step 2: Install dependencies

```bash
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y
pip install pytorch-lightning wandb matplotlib
```

### Step 3: WandB Setup (optional but recommended)

```bash
wandb login
```

## Running Experiments

### Part A: Train CNN from Scratch

```bash
cd partA
python scripts/train.py \
    --num_filters=32 \
    --activation=ReLU \
    --filter_organisation=same \
    --data_augmentation=True \
    --use_batchnorm=False \
    --dropout_rate=0.2 \
    --batch_size=32 \
    --max_epochs=10 \
    --lr=1e-3
```

Run hyperparameter sweep:

```bash
wandb sweep scripts/sweep_config.yaml
wandb agent <entity>/da6401_assignment02/<sweep_id>
```

Generate predictions:

```bash
python scripts/show_predictions.py
```

### Part B: Fine-tune ResNet50

```bash
cd partB
python scripts/train_finetune.py \
    --unfreeze_layers=2 \
    --batch_size=32 \
    --max_epochs=10 \
    --lr=1e-4
```

Run hyperparameter sweep:

```bash
wandb sweep scripts/sweep_config.yaml
wandb agent <entity>/da6401_assignment02/<sweep_id>
```

Generate predictions:

```bash
python scripts/show_predictions.py
```

## Logging and Monitoring

- **GPU Monitoring:**

```bash
watch -n1 nvidia-smi
```

- **Weights & Biases Dashboards:**
  - [Report](https://wandb.ai/)

## Best Practices

- Ensure consistent train, validation, and test splits.
- Regularly commit progress to GitHub.
- Document hyperparameter choices and experimental outcomes clearly.
