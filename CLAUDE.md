# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

LostPaw is a contrastive learning transformer system for finding lost pets through facial recognition. It follows a standard Python package structure with:
- **`lostpaw/`**: Core library package containing ML pipeline components
- **`scripts/`**: Executable command-line tools that use the lostpaw package
- Docker-based development environment with GPU support

## Project Structure

### `lostpaw/` - Core Library Package
```
lostpaw/
├── config/          # Configuration management
│   ├── args.py      # Command-line argument parsing with YAML support
│   └── config.py    # Configuration classes (TrainConfig, OptimizerConfig)
├── configs/         # YAML configuration files
│   ├── default.yaml # Default training configuration
│   ├── sweep.yaml   # Hyperparameter sweep configuration
│   └── container.yaml # Container configuration
├── data/            # Data processing and loading
│   ├── auto_augment.py    # Data augmentation utilities
│   ├── data_folder.py     # Folder-based data handling
│   ├── dataset.py         # Dataset classes for contrastive learning
│   └── extract_pets.py    # Pet face extraction utilities
└── model/           # Model definitions and training
    ├── loss.py      # Contrastive loss functions
    ├── model.py     # PetViTContrastiveModel architecture
    └── trainer.py   # Training loop implementation
```

### `scripts/` - Executable Scripts
- **Training & Evaluation**: `train.py`, `test.py`, `sweep.py`
- **Data Processing**: `extract_pets.py`, `clean_dataset.py`, `pick_broken_image.py`
- **Dataset Generation**: `convert_petface.py`, `generate_dogfacenet_data.py`
- **Inference & Utilities**: `inference_server.py`, `predict_pair.py`, `check_model.py`
- **Visualization**: `visualize_data.py`, `plot_test_scores.py`

## Common Development Commands

### Environment Setup
```bash
# Build and run Docker container with GPU support
docker buildx build --load -t lostpaw-transformer .
docker run -it --gpus all --dns 8.8.8.8 --name lostpaw-transformer-dev -v $(pwd):/app -w /app lostpaw-transformer bash

# Install Python package in development mode
pip install -e .
pip install "wandb==0.15.12" "pydantic<2.0"
```

### Data Preparation Pipeline
```bash
# 1. Convert PetFace dataset
python scripts/convert_petface.py input_folder output_folder

# 2. Clean dataset (remove duplicates)
python scripts/clean_dataset.py output/raw-data --deduplicate

# 3. Remove broken images
python scripts/pick_broken_image.py output/raw-data

# 4. Generate JSONL data file
python scripts/generate_dogfacenet_data.py

# 5. Extract pet faces with augmentation (multi-threaded)
python scripts/extract_pets.py \
  --info_file ./output/raw-data/raw-data.jsonl \
  --model_path ./output/weights \
  --output_dir ./output/generated \
  --threads 4 --batch_size 4

# 6. Merge multi-threaded output
python scripts/extract_pets_merge.py output/generated/thread_* output/data
```

### Model Training & Testing
```bash
# Train model with cross-validation
CUDA_VISIBLE_DEVICES=0 python scripts/train.py -c lostpaw/configs/default.yaml

# Evaluate model
python scripts/test.py -c lostpaw/configs/default.yaml

# Hyperparameter sweep
python scripts/sweep.py --model ./models --info ./data/train.data --config lostpaw/configs/sweep.yaml

# Compare two images
python scripts/predict_pair.py image1.jpg image2.jpg --threshold 0.85
```

### Inference & Utilities
```bash
# Start Flask inference server
python scripts/inference_server.py --model ../output/models/model.pt

# Visualize dataset
python scripts/visualize_data.py output/data

# Check model integrity
python scripts/check_model.py

# Plot training metrics
python scripts/plot_test_scores.py
```

## Key Components

### Configuration System
- Hierarchical YAML-based configuration with command-line overrides
- Base config: `lostpaw/configs/default.yaml`
- Override: `python scripts/train.py -c custom_config.yaml --param_name value`

### Model Architecture
- Vision Transformer (ViT) backbone with contrastive learning
- Uses Timm and HuggingFace Transformers
- Configurable output dimensions and loss margins
- Supports model checkpointing and resuming

### Data Pipeline
1. Raw images → DETR-based pet face extraction
2. Face crops → Data augmentation (rotation, flip, color jitter)
3. Augmented data → JSONL format for pair-based training
4. Contrastive learning with positive/negative pairs

### Import Patterns
```python
# Configuration
from lostpaw.config import TrainConfig, get_args

# Model components
from lostpaw.model import PetViTContrastiveModel, PetContrastiveLoss
from lostpaw.model.trainer import Trainer

# Data handling
from lostpaw.data import PetImageDataset, DetrPetExtractor, RandomPairDataset
```

## Important Notes
- Always use Docker container for consistent environment
- GPU required for training (CUDA support needed)
- Scripts should be run from project root directory
- Model achieves 88.8% F1-score on cross-validation
- All path references in moved scripts use `../` prefix for output directory
- Package follows pip installable structure (`pip install -e .`)