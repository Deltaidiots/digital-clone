# Data Flow and Processing Pipeline

## Overview

This document details how data flows through the SyncTalk_2D system, from raw inputs to final video output, covering both training and inference pipelines.

## Training Data Flow

### 1. Raw Input Processing

```
Raw Video File (dataset/name/name.mp4)
    ↓
Video Analysis & Preprocessing
    ├── Frame Extraction (25 FPS)
    ├── Audio Extraction
    ├── Face Detection
    └── Landmark Detection
    ↓
Processed Dataset Structure
```

### 2. Dataset Structure Creation

The preprocessing creates this structure:
```
dataset/name/
├── name.mp4                    # Original video
├── full_body_img/              # Extracted frames
│   ├── 0.jpg
│   ├── 1.jpg
│   └── ...
├── landmarks/                  # Facial landmarks
│   ├── 0.lms
│   ├── 1.lms
│   └── ...
├── aud.wav                     # Extracted audio
└── aud_ave.npy                # Preprocessed audio features
```

### 3. Training Data Pipeline

```python
# Pseudocode for training data flow
for each_frame in video:
    # Image processing
    img = load_image(frame_path)
    landmarks = load_landmarks(lms_path)
    
    # Face cropping based on landmarks
    face_crop = crop_face_region(img, landmarks)
    face_crop = resize(face_crop, (328, 328))
    
    # Create masked version
    masked_crop = mask_lower_face(face_crop)
    input_tensor = concatenate([face_crop, masked_crop])  # 6 channels
    
    # Audio processing
    audio_features = get_audio_features(audio_data, frame_index)
    
    # Target is the unmasked face crop
    target = face_crop[4:324, 4:324]  # 320x320 region
    
    yield (input_tensor, audio_features, target)
```

## Inference Data Flow

![Inference Flow](diagrams/inference-flow.puml)

### 1. Input Preparation

```
User Inputs:
├── Character name (--name)
├── Audio file (--audio_path)
└── Optional parameters
    ↓
System Preparation:
├── Load character checkpoint
├── Load template images/landmarks
└── Process input audio
```

### 2. Audio Processing Pipeline

```python
# Audio processing for inference
audio_file → AudioDataset → DataLoader → AudioEncoder → features

# Detailed steps:
1. Load audio file (.wav)
2. Convert to mel spectrogram
3. Batch process through AudioEncoder
4. Extract frame-aligned features
5. Add padding for sequence boundaries
```

### 3. Frame Generation Loop

```python
for frame_index in range(total_frames):
    # 1. Template selection (bouncing between frames)
    template_frame = select_template(frame_index, template_range)
    
    # 2. Load template and landmarks
    img = load_template(template_frame)
    landmarks = load_landmarks(template_frame)
    
    # 3. Face region extraction
    xmin, ymin, xmax, ymax = calculate_face_bounds(landmarks)
    face_crop = img[ymin:ymax, xmin:xmax]
    
    # 4. Prepare model input
    face_328 = resize(face_crop, (328, 328))
    face_320 = face_328[4:324, 4:324]  # Inner region
    masked_320 = apply_mouth_mask(face_320)
    
    # 5. Create 6-channel input
    input_tensor = concatenate([
        face_320,      # Original (masked)
        masked_320     # Fully masked
    ])
    
    # 6. Get corresponding audio features
    audio_feat = get_audio_features(audio_features, frame_index)
    
    # 7. Model inference
    generated_320 = model(input_tensor, audio_feat)
    
    # 8. Reconstruct full image
    face_328[4:324, 4:324] = generated_320
    reconstructed_crop = resize(face_328, original_crop_size)
    
    # 9. Paste back to full frame
    img[ymin:ymax, xmin:xmax] = reconstructed_crop
    
    # 10. Write to video
    video_writer.write(img)
```

## Audio Feature Processing

### Feature Extraction Types

1. **AVE (Audio-Visual Encoder)**
   ```python
   # Processing pipeline
   mel_spectrogram → AudioEncoder → feature_vector
   # Output shape after reshaping: (32, 16, 16)
   ```

2. **Hubert**
   ```python
   # More detailed speech features
   audio → HubertModel → feature_vector
   # Output shape: (32, 32, 32)
   ```

3. **WeNet**
   ```python
   # Speech recognition features
   audio → WeNetModel → feature_vector
   # Output shape: (256, 16, 32)
   ```

### Temporal Alignment

```python
def get_audio_features(audio_feats, frame_index):
    """Extract audio features for specific frame with context"""
    # Add context from previous and next frames
    start_idx = max(0, frame_index - 1)
    end_idx = min(len(audio_feats), frame_index + 2)
    
    context_features = audio_feats[start_idx:end_idx]
    
    # Reshape based on audio encoder type
    if mode == "ave":
        return context_features.reshape(32, 16, 16)
    elif mode == "hubert":
        return context_features.reshape(32, 32, 32)
    elif mode == "wenet":
        return context_features.reshape(256, 16, 32)
```

## Video Processing Pipeline

### Template Management

```python
# Bouncing template selection for natural movement
def select_template_frame(current_index, max_frames):
    if direction == "forward":
        template_idx += 1
        if template_idx >= max_frames - 1:
            direction = "backward"
    else:  # backward
        template_idx -= 1
        if template_idx <= 0:
            direction = "forward"
    
    return template_idx
```

### Face Region Processing

```python
def process_face_region(img, landmarks):
    # Calculate face bounds from landmarks
    xmin = landmarks[1][0]   # Left face boundary
    ymin = landmarks[52][1]  # Top of mouth region
    xmax = landmarks[31][0]  # Right face boundary
    
    # Make square crop
    width = xmax - xmin
    ymax = ymin + width
    
    # Extract and resize
    crop = img[ymin:ymax, xmin:xmax]
    return resize(crop, (328, 328))
```

## Output Video Generation

### Frame Assembly

```python
# Final video creation pipeline
1. Generate all frames → temp_video.mp4
2. Combine with original audio → ffmpeg
3. Apply quality settings (CRF)
4. Save final output
```

### Quality Control

```python
# Video encoding parameters
if enhancement_enabled:
    crf_value = 18  # Higher quality
else:
    crf_value = 20  # Standard quality

ffmpeg_cmd = f"""
ffmpeg -i {temp_video} -i {audio_file} 
       -c:v libx264 -c:a aac 
       -crf {crf_value} 
       {output_video} -y
"""
```

## Memory Management

### Batch Processing
- Audio features processed in batches of 64
- Video frames processed individually to manage memory
- Automatic garbage collection between frames

### GPU Memory Optimization
```python
with torch.no_grad():  # Disable gradient computation
    prediction = model(input_tensor, audio_features)
    
# Move to CPU immediately to free GPU memory
prediction = prediction.cpu().numpy()
```

## Error Handling and Fallbacks

### Graceful Degradation
- Missing landmarks → skip frame or use previous
- Audio processing errors → use silence features
- Model errors → return original frame
- File I/O errors → detailed logging and recovery

### Data Validation
- Image dimension checks
- Audio length validation
- Landmark format verification
- Model output sanity checks

This pipeline ensures robust, high-quality video generation while maintaining real-time performance capabilities.
