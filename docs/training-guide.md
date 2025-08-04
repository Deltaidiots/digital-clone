# Training Guide

## Overview

This document provides a comprehensive guide to training SyncTalk_2D models, including data preparation, training configuration, and optimization strategies.

## Prerequisites

### Hardware Requirements
- **GPU**: NVIDIA GPU with 8GB+ VRAM (recommended: RTX 3080 or better)
- **RAM**: 32GB+ system RAM
- **Storage**: 50GB+ free space for dataset and checkpoints
- **CPU**: Multi-core processor for data preprocessing

### Software Requirements
- Python 3.10
- PyTorch 2.2.0
- CUDA 12.1
- FFmpeg
- OpenCV

## Data Preparation

### Recording Guidelines

1. **Video Duration**: 5+ minutes of speaking content
2. **Resolution**: 1080p or higher recommended
3. **Frame Rate**: Any (will be converted to 25fps)
4. **Lighting**: Consistent, well-lit environment
5. **Background**: Static, unchanging background
6. **Subject Position**: Head facing camera, minimal movement
7. **Clothing**: Avoid highly textured clothing, prefer solid colors
8. **Audio**: Clear speech, no background voices

### Recording Setup Checklist

```markdown
□ Camera fixed in position (no movement during recording)
□ Consistent lighting throughout recording
□ 5-second silence at beginning and end
□ Clear audio without background noise
□ Subject maintains consistent position
□ No other people visible or audible
□ Avoid reflective jewelry or glasses glare
```

### Data Processing Pipeline

1. **Place Video File**:
   ```bash
   mkdir -p dataset/YOUR_NAME
   cp your_video.mp4 dataset/YOUR_NAME/YOUR_NAME.mp4
   ```

2. **Run Preprocessing**:
   ```bash
   bash training_328.sh YOUR_NAME 0
   ```

This creates the following structure:
```
dataset/YOUR_NAME/
├── YOUR_NAME.mp4           # Original video
├── YOUR_NAME_25fps.mp4     # Converted to 25fps
├── aud.wav                 # Extracted audio
├── aud_ave.npy            # Processed audio features
├── full_body_img/         # Extracted frames
│   ├── 0.jpg
│   ├── 1.jpg
│   └── ...
└── landmarks/             # Facial landmarks
    ├── 0.lms
    ├── 1.lms
    └── ...
```

## Training Configuration

### Key Parameters

```python
# In train_328.py
batch_size = 8          # Reduce if OOM occurs
learning_rate = 1e-4    # Adam optimizer learning rate
num_epochs = 200        # Total training epochs
save_interval = 10      # Checkpoint saving frequency
```

### Audio Encoder Options

1. **AVE (Default)**:
   ```bash
   python train_328.py --name YOUR_NAME --asr ave
   ```
   - Best balance of quality and training speed
   - Recommended for most users

2. **Hubert**:
   ```bash
   python train_328.py --name YOUR_NAME --asr hubert
   ```
   - Better speech understanding
   - Longer training time

3. **WeNet**:
   ```bash
   python train_328.py --name YOUR_NAME --asr wenet
   ```
   - Detailed phoneme features
   - Highest memory requirements

## Training Process

![Training Pipeline](diagrams/training-pipeline.puml)

### Automated Training Script

```bash
# Basic training
bash training_328.sh YOUR_NAME 0

# With specific GPU
bash training_328.sh YOUR_NAME 1

# The script performs:
# 1. Video preprocessing
# 2. Audio feature extraction
# 3. Face detection and landmark extraction
# 4. Model training
# 5. Checkpoint saving
```

### Manual Training Steps

1. **Data Preprocessing**:
   ```bash
   cd data_utils
   python process.py --input_video ../dataset/YOUR_NAME/YOUR_NAME.mp4 \
                     --output_dir ../dataset/YOUR_NAME
   ```

2. **Start Training**:
   ```bash
   python train_328.py --name YOUR_NAME --asr ave
   ```

3. **Monitor Progress**:
   ```bash
   # Check GPU usage
   nvidia-smi
   
   # Monitor training logs
   tail -f training.log
   ```

## Training Monitoring

### Key Metrics

1. **Loss Curve**: Should decrease steadily
2. **GPU Utilization**: Should be 80-90%
3. **Memory Usage**: Monitor VRAM usage
4. **Training Speed**: ~1-2 seconds per batch

### Checkpoint Management

Checkpoints are saved every 10 epochs:
```
checkpoint/YOUR_NAME/
├── 10.pth
├── 20.pth
├── 30.pth
└── ...
```

### Early Stopping Criteria

Consider stopping training if:
- Loss plateaus for 50+ epochs
- Visual quality stops improving
- Overfitting occurs (training loss << validation loss)

## Common Issues and Solutions

### Out of Memory (OOM)

**Symptoms**: CUDA out of memory error

**Solutions**:
1. Reduce batch size:
   ```python
   batch_size = 4  # or even 2
   ```

2. Use gradient accumulation:
   ```python
   accumulation_steps = 2
   effective_batch_size = batch_size * accumulation_steps
   ```

3. Enable mixed precision:
   ```python
   from torch.cuda.amp import autocast, GradScaler
   scaler = GradScaler()
   ```

### Slow Training

**Symptoms**: Very slow progress, low GPU utilization

**Solutions**:
1. Check data loading:
   ```python
   num_workers = 4  # Increase dataloader workers
   pin_memory = True
   ```

2. Optimize preprocessing:
   ```python
   # Cache processed data
   # Use faster image loading
   ```

### Poor Quality Results

**Symptoms**: Blurry or unrealistic outputs

**Solutions**:
1. Check input data quality
2. Increase training duration
3. Adjust learning rate:
   ```python
   learning_rate = 5e-5  # Lower for stability
   ```

4. Use learning rate scheduling:
   ```python
   scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 
                                               step_size=50, 
                                               gamma=0.5)
   ```

## Training Best Practices

### Data Quality
- Use high-quality source video
- Ensure consistent lighting
- Minimize background distractions
- Record diverse speech content

### Model Training
- Start with default parameters
- Monitor training curves
- Save checkpoints frequently
- Test inference during training

### Resource Management
- Use appropriate batch size for your GPU
- Monitor system resources
- Clean up old checkpoints to save space
- Use efficient data loading

## Advanced Training Options

### Custom Loss Functions

```python
# Add perceptual loss for better quality
class PerceptualLoss(nn.Module):
    def __init__(self):
        super().__init__()
        # Implementation details...

# Combined loss
total_loss = l1_loss + 0.1 * perceptual_loss
```

### Data Augmentation

```python
# Add augmentation during training
transforms = [
    RandomBrightness(0.1),
    RandomContrast(0.1),
    ColorJitter(0.05),
]
```

### Multi-GPU Training

```python
# Use DataParallel for multiple GPUs
model = nn.DataParallel(model)

# Or DistributedDataParallel for better performance
model = nn.parallel.DistributedDataParallel(model)
```

## Training Timeline

### Typical Training Schedule

- **Hours 0-1**: Setup and data preprocessing
- **Hours 1-2**: Initial training, loss drops rapidly
- **Hours 2-4**: Steady improvement in quality
- **Hours 4-5**: Fine-tuning, diminishing returns
- **Total**: ~5 hours for good quality model

### Checkpoint Selection

Choose the best checkpoint based on:
1. Visual quality of sample outputs
2. Lip synchronization accuracy
3. Overall stability
4. Generalization to new audio

The final trained model will be ready for inference and can generate high-quality talking head videos from any audio input.
