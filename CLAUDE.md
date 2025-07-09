# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

LostPaw is a contrastive learning transformer system for finding lost pets through facial recognition. It consists of:
- A PyTorch-based ML pipeline for training pet face recognition models
- Data processing and augmentation utilities

## Common Development Commands

### Environment Setup
```bash
# Build and run Docker container with GPU support
docker buildx build --load -t lostpaw-transformer .
docker run -it --gpus all --dns 8.8.8.8 --name lostpaw-transformer-dev -v $(pwd):/app -w /app lostpaw-transformer bash

# Install Python package in development mode
pip install -e .
```

### Model Training & Testing
```bash
# Train model
CUDA_VISIBLE_DEVICES=0 python scripts/train.py -c lostpaw/configs/default.yaml

# Run tests
python scripts/test.py -c lostpaw/configs/default.yaml

# Run inference server
python scripts/inference_server.py
```

### Data Processing
```bash
# Extract pet faces with augmentation
python scripts/extract_pets.py --info_file ./output/raw-data/raw-data.jsonl --model_path ./output/weights --output_dir ./output/generated --threads 4 --batch_size 4

# Merge multi-threaded output
python scripts/extract_pets_merge.py output/generated/thread_* output/data
```

## Architecture & Key Components

### Python ML Pipeline (`/lostpaw/`)
- **Configuration**: YAML-based config system in `/lostpaw/configs/`
- **Model**: Contrastive learning transformer using Timm and HuggingFace
- **Training**: Supports Weights & Biases logging, configurable loss margins
- **Data**: Custom dataset classes with augmentation support

### Key Files to Understand
- `lostpaw/model/model.py`: Core transformer architecture
- `lostpaw/model/trainer.py`: Training loop implementation

### Configuration System
The project uses a hierarchical configuration system:
1. Base config: `lostpaw/configs/default.yaml`
2. Override via command line: `-c path/to/config.yaml`
3. Individual parameters: `--param_name value`

### Data Pipeline
1. Raw images → Pet face extraction (using pretrained model)
2. Face crops → Augmentation (rotation, flip, color jitter)
3. Augmented data → JSONL format for training
4. Training uses contrastive loss with configurable margins

## Important Notes
- Always use Docker container for consistent environment
- GPU required for training (CUDA support needed)
- Model achieves 88.8% F1-score on cross-validation