# SyncTalk_2D Architecture

This document provides a detailed overview of the SyncTalk_2D architecture, from high-level components to the specifics of the model and data flow.

## High-Level Overview

![System Overview](diagrams/system-overview.puml)

SyncTalk_2D is a deep learning-based system designed to generate realistic 2D talking head animations from an audio source and a single static image of a person. The core of the system is a U-Net-based generator that synthesizes lip movements synchronized with the input audio.

The architecture is modular and consists of several key components:

- **Audio Processing**: An audio encoder extracts meaningful features from the input audio waveform.
- **Video Processing**: The system uses a template image of a person and extracts landmarks to define the facial structure.
- **Core Model (U-Net)**: A U-Net model takes the audio features and a masked template image to generate a new image with synchronized lip movements.
- **Inference Pipeline**: Combines the components to generate a full video sequence frame by frame.
- **Training Pipeline**: Trains the U-Net model on a dataset of video and corresponding audio.
- **API and Web Interface**: A FastAPI server provides an easy-to-use interface for generating animations.

## Model Architecture

The core of SyncTalk_2D is a **single-stage generative model** based on the U-Net architecture. This is not a multi-stage pipeline where different models handle different parts of the generation (e.g., one for landmarks, one for texture). Instead, a single, powerful U-Net model handles the entire image synthesis process.

### U-Net Generator (`unet_328.py`)

The generator is an encoder-decoder network with skip connections, which is characteristic of a U-Net.

- **Input**: The model takes two main inputs:
    1.  **Concatenated Image Tensor**: This consists of a real face image (with the lower half of the face masked out) and a completely masked version of the same image. This provides the model with the upper part of the face for context and a canvas to generate the lower part.
    2.  **Audio Feature Tensor**: Processed audio features that guide the generation of the mouth shapes.

- **Architecture**:
    - **Encoder**: A series of convolutional layers that downsample the input image, capturing high-level features.
    - **Bottleneck**: The audio features are injected at the bottleneck of the U-Net. The audio features are processed through several convolutional layers and then reshaped to match the spatial dimensions of the image features at the bottleneck. This is the critical step where the audio signal modulates the video generation.
    - **Decoder**: A series of transposed convolutional layers that upsample the features back to the original image resolution, reconstructing the face with the correct lip sync.
    - **Skip Connections**: Connections between corresponding encoder and decoder layers allow the model to reuse low-level features from the input image (like skin texture, lighting), leading to higher-quality and more realistic outputs.

- **Output**: The model outputs a 320x320 pixel image representing the generated lower half of the face. This generated region is then pasted back into the original high-resolution frame.

## Data Flow

### Audio Processing

1.  **Loading**: The input audio file (e.g., a `.wav` file) is loaded.
2.  **Feature Extraction (`utils.py`)**: An `AudioEncoder` model is used to extract deep features from the audio. This is not a simple spectrogram; it's a learned representation. The `AudioEncoder` is a pre-trained model that has learned to extract features relevant for audio-visual synchronization.
3.  **Chunking**: For inference, the audio features are processed in small, overlapping chunks corresponding to the video frame rate. This ensures that each generated frame is conditioned on the correct segment of audio.

### Video Processing (Inference)

1.  **Template Image**: A single high-resolution image of the target person is used as a template.
2.  **Landmark Detection**: Facial landmarks are pre-extracted for the template video frames. During inference, the system uses these landmarks to identify the face region.
3.  **Cropping and Resizing**: For each frame to be generated, the face region is cropped from the template image based on the landmarks. The crop is specifically focused on the mouth region. This cropped image is then resized to the resolution the model was trained on (328x328).
4.  **Masking**: The lower part of the face in the cropped image is masked out (set to black). This is the area the U-Net will fill in.
5.  **Model Input**: The masked image is fed into the U-Net, along with the corresponding audio features for that time step.
6.  **Pasting Back**: The generated 320x320 output from the model is placed back into the 328x328 cropped image, and then the full crop is resized back to its original dimensions and pasted into the template frame.
7.  **Video Creation**: The sequence of generated frames is compiled into a final video file using `ffmpeg`.

## Training Process (`train_328.py`)

The training process is designed to teach the U-Net model how to generate realistic lip movements that are synchronized with the audio.

1.  **Dataset Preparation (`process.py`)**:
    - A video of the target person speaking is processed.
    - The video is split into frames.
    - Audio is extracted from the video.
    - Facial landmarks are detected and saved for each frame.
    - The data is organized into a specific directory structure.

2.  **Training Resolution**: The model is trained on **328x328** pixel crops of the face. The core generation happens on a 320x320 region within this crop. This is a key improvement for generating high-definition results.

3.  **Training Loop**:
    - The `train_328.py` script runs the training loop.
    - **Data Loading**: A custom `Dataset` class loads pairs of (video frame, audio segment).
    - **Batching**: The data is fed into the model in batches.
    - **Forward Pass**: For each batch, the model performs a forward pass, generating a predicted face image from the masked input and audio features.
    - **Loss Calculation**: The loss is calculated between the model's generated image and the ground truth (the actual, unmasked video frame). A common loss function for this type of task is L1 or L2 loss, which measures the pixel-wise difference.
    - **Backpropagation**: The loss is backpropagated through the network to compute gradients.
    - **Optimization**: An optimizer (like Adam) updates the model's weights based on the gradients.
    - **Checkpointing**: The model's state is periodically saved to a checkpoint file (`.pth`). This allows training to be resumed and allows the trained model to be used for inference.

This entire process results in a model that can take an arbitrary audio file and a single image of the trained person and generate a high-quality, lip-synced video.
