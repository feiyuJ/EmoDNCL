# EmoDNCL: Dual-stream Negative-sample-free Contrastive Learning with Neurophysiological Augmentation for EEG Emotion Recognition

A PyTorch implementation of DNCL model for EEG-based emotion recognition.

## Features
- Self-distillation learning framework
- Self-organized graph neural networks
- Two-stage training (pre-training + fine-tuning)
- Support for SEED and SEEDIV datasets
- Sensitive channel-aware data augmentation
- Neighbor-based EEG signal enhancement

## Model Architecture
- **DNCL**: Teacher-student self-distillation framework
- **SOGNN**: Self-organized graph construction and convolution
- **Two-stage training**: Unsupervised pre-training followed by supervised fine-tuning

## Installation

### Requirements
- Python 3.7+
- PyTorch 1.8.0+
- PyTorch Geometric 2.0.0+

### Install Dependencies
```bash
pip install -r requirements.txt
```

## Data Preparation

Organize your data as follows:
```
data/
├── SEED/
│   └── data_100per/processed/
│       ├── V_1_Train_CV15_0.dataset
│       └── ...
└── SEEDIV/
    └── data_100per/processed/
        ├── V_1_Train_CV15_0.dataset
        └── ...
```

## Data Augmentation

The model uses NINA (Neighbor-based Intelligent Neural Augmentation):

- **Neighbor-based enhancement**: Uses channel neighbor relationships from `datasets/neighbors_fixed_full.csv`
- **Time-selective enhancement**: Applies augmentation to selected time segments
- **Value replacement**: Replaces channel values with neighbor max/min values
- **Random channel selection**: Randomly selects channels for augmentation

### Augmentation Parameters

- `--aug1_n`: Number of channels to augment for aug1 (default: 9)
- `--aug2_n`: Number of channels to augment for aug2 (default: 6)
- `--aug1_p`: Replacement ratio for aug1 (default: 0.5 = 50% original + 50% neighbor value)
- `--aug2_p`: Replacement ratio for aug2 (default: 0.5 = 50% original + 50% neighbor value)
- `--aug1_r`: Time segment ratio for aug1 (default: 0.3)
- `--aug2_r`: Time segment ratio for aug2 (default: 0.3)

### NINA Augmentation Strategy

- **Random channel selection**: Randomly selects channels for augmentation
- **Neighbor-based replacement**: Uses neighbor channel values for enhancement
- **Time-selective**: Only applies to selected time segments
- **No random seeds**: Ensures different augmentation patterns each time

## Usage

### Pre-training (Self-supervised)
```bash
python main.py \
    --training_mode self_supervised \
    --selected_dataset SEEDIV \
    --epochs 400 \
    --gpu_id 0 \
    --batch_size 16 \
    --aug1_p 0.5 \
    --aug1_r 0.3 \
    --aug2_p 0.5 \
    --aug2_r 0.3
```

### Fine-tuning (Supervised)
```bash
python main.py \
    --training_mode FT \
    --selected_dataset SEEDIV \
    --epochs 300 \
    --gpu_id 0 \
    --batch_size 16 \
    --aug1_p 0.5 \
    --aug1_r 0.3 \
    --aug2_p 0.5 \
    --aug2_r 0.3
```

### Custom NINA Augmentation Parameters
```bash
python main.py \
    --training_mode self_supervised \
    --selected_dataset SEEDIV \
    --epochs 400 \
    --gpu_id 0 \
    --aug1_n 12 \
    --aug2_n 8 \
    --aug1_p 0.7 \
    --aug1_r 0.4 \
    --aug2_p 0.6 \
    --aug2_r 0.5
```

## Arguments

| Argument | Description | Default | Options |
|----------|-------------|---------|---------|
| `--training_mode` | Training mode | FT | self_supervised, FT |
| `--selected_dataset` | Dataset name | SEEDIV | SEED, SEEDIV |
| `--epochs` | Training epochs | 300 | int |
| `--gpu_id` | GPU device ID | 0 | int |
| `--batch_size` | Batch size | 16 | int |
| `--aug1_n` | Channels to augment for aug1 | 9 | int |
| `--aug2_n` | Channels to augment for aug2 | 6 | int |
| `--aug1_p` | Replacement ratio for aug1 | 0.5 | float (0-1) |
| `--aug2_p` | Replacement ratio for aug2 | 0.5 | float (0-1) |
| `--aug1_r` | Time ratio for aug1 | 0.3 | float (0-1) |
| `--aug2_r` | Time ratio for aug2 | 0.3 | float (0-1) |
| `--experiment_description` | Experiment name | DNCL_Experiment | string |

## Training Settings

### SEEDIV Dataset
- Pre-training: 400 epochs
- Fine-tuning: 300 epochs
- Learning rate: 0.00001
- Optimizer: Adam
- Batch size: 16

### SEED Dataset
- Pre-training: 300 epochs
- Fine-tuning: 300 epochs
- Learning rate: 0.00001
- Optimizer: Adam
- Batch size: 16

## Results

Results are saved in:
- Models: `./model/{dataset}/{experiment}/saved_models/`
- Logs: `./result/{dataset}/{experiment}/`

## Project Structure
```
GITHUB/
├── models/
│   ├── dncl.py              # DNCL model implementation
│   └── __init__.py
├── data/
│   ├── loader.py            # Data loading
│   ├── augmentation.py      # Data augmentation
│   └── __init__.py
├── main.py                   # Training script
├── requirements.txt          # Dependencies
└── README.md                 # This file
```

## Citation



## License

This project is licensed under the MIT License - see the LICENSE file for details.
