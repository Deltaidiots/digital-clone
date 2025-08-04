# SyncTalk_2D

SyncTalk_2D is a 2D lip-sync video generation model based on SyncTalk and [Ultralight-Digital-Human](https://github.com/anliyuan/Ultralight-Digital-Human). It can generate lip-sync videos with high quality and low latency, and it can also be used for real-time lip-sync video generation.

Compared to the Ultralght-Digital-Human, we have improved the audio feature encoder and increased the resolution to 328 to accommodate higher-resolution input video. This version can realize high-definition, commercial-grade digital humans.

与Ultralght-Digital-Human相比，我们改进了音频特征编码器，并将分辨率提升至328以适应更高分辨率的输入视频。该版本可实现高清、商业级数字人。

## Architecture

SyncTalk_2D uses a modular architecture following SOLID design principles:

- **Single Responsibility**: Each component (model service, TTS service, frame generation) has a distinct responsibility
- **Open/Closed**: Components are extensible without requiring modification (e.g., the TTSService interface)
- **Liskov Substitution**: Derived classes (like the enhanced frame generator) can be substituted for their base classes
- **Interface Segregation**: Clean interfaces between components with minimal dependencies
- **Dependency Inversion**: High-level modules depend on abstractions, not concrete implementations

The system is divided into these main components:

1. **ModelService**: Loads and manages neural network models (U-Net and Audio Encoder)
2. **TTSService**: Handles text-to-speech synthesis with graceful fallback mechanisms
3. **FrameGenerationService**: Generates animated frames from text or audio input
4. **API Layer**: Exposes functionality through FastAPI endpoints with proper error handling

## Setting up
Set up the environment
``` bash
conda create -n synctalk_2d python=3.10
conda activate synctalk_2d
```
``` bash
# install dependencies
conda install pytorch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 pytorch-cuda=12.1 -c pytorch -c nvidia
conda install -c conda-forge ffmpeg  #very important
pip install opencv-python transformers soundfile librosa onnxruntime-gpu configargparse
pip install numpy==1.23.5
pip install fastapi uvicorn pydantic python-multipart piper-tts
```

## Prepare your data
1. Record a 5-minute video with your head facing the camera and without significant movement. At the same time, ensure that the camera does not move and the background light remains unchanged during video recording.
2. Don't worry about FPS, the code will automatically convert the video to 25fps.
3. No second person's voice can appear in the recorded video, and a 5-second silent clip is left at the beginning and end of the video.
4. Don't wear clothes with overly obvious texture, it's better to wear single-color clothes.
5. The video should be recorded in a well-lit environment.
6. The audio should be clear and without background noise.


## Train
1. put your video in the 'dataset/name/name.mp4' 

- example: dataset/May/May.mp4

2. run the process and training script

``` bash
bash training_328.sh name gpu_id
```
- example: bash training_328.sh May 0

- Waiting for training to complete, approximately 5 hours

- If OOM occurs, try reducing the size of batch_size

## Inference

``` bash
python inference_328.py --name data_name --audio_path path_to_audio.wav
```
- example: python inference_328.py --name May --audio_path demo/talk_hb.wav

- the result will be saved in the 'result' folder

## Web Interface & API

SyncTalk_2D now includes a web interface and API for real-time talking head animation. This allows you to generate animations through a browser using either text input (which gets converted to speech) or by uploading audio files directly.

### Starting the Web Server

```bash
python api.py
```

This will start a FastAPI server on port 8000. You can access the web interface by visiting `http://localhost:8000` in your browser.

### Features

- **Text-to-Speech Animation**: Enter text and watch the avatar speak it in real-time
- **Audio File Animation**: Upload a WAV file (16kHz, 16-bit PCM recommended) and see the avatar speak along with it
- **Streaming API**: All animations are streamed frame-by-frame for real-time viewing
- **Debug Mode**: Save generated frames and audio for troubleshooting and optimization
- **Audio-Video Synchronization**: Precise timing to ensure lips match the audio perfectly

### API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Web interface |
| `/generate` | GET | Generate animation from text (streaming) |
| `/generate` | POST | Generate animation from text (JSON body) |
| `/generate-from-audio` | POST | Generate animation from uploaded audio |
| `/avatar` | GET | Get a static avatar frame |
| `/get-tts-audio` | GET | Generate and download TTS audio |
| `/health` | GET | Check service health |
| `/debug/status` | GET | Get current debug settings |
| `/debug/config` | POST | Update debug settings |

### Configuration

The application uses a centralized configuration system defined in `config.py`:

```python
from config import settings

# Access configuration
device = settings.models.device
tts_model_path = settings.tts.model_path
```

Configuration sections include:
- **models**: Neural network models and device settings
- **tts**: Text-to-speech service configuration
- **debug**: Debug and diagnostics settings

### Debug Features

SyncTalk_2D includes debug features to help troubleshoot issues:

1. **Save Frames**: When enabled, each generated frame will be saved to the `debug/frames` directory
2. **Save Audio**: When enabled, generated or uploaded audio will be saved to the `debug/audio` directory
3. **Chunked Processing**: The system now processes audio in smaller chunks to enable real-time streaming

To enable debug features:

1. Use the debug panel in the web interface (click on "Debug Controls" at the bottom)
2. Or update settings via API: `POST /debug/config` with JSON body:
   ```json
   {
     "enabled": true,
     "save_frames": true,
     "save_audio": true
   }
   ```

## Code Structure

```
├── api.py                       # FastAPI web server and endpoints
├── config.py                    # Central configuration management
├── docs/                        # Documentation
│   ├── architecture.md          # System architecture overview
│   ├── model-details.md         # Detailed model architecture and specifications
│   ├── data-flow.md            # Data processing pipelines and flow
│   ├── training-guide.md       # Comprehensive training guide
│   └── api-reference.md        # API endpoints and usage
├── services/
│   ├── models.py                # Neural network model management
│   ├── tts.py                   # Text-to-speech synthesis
│   └── generator.py             # Frame generation and animation
├── static/
│   └── index.html               # Web interface
├── utils.py                     # Utility functions
├── unet_328.py                  # U-Net neural network architecture
├── syncnet_328.py               # SyncNet for audio-visual synchronization
├── inference_328.py             # Offline inference script
├── train_328.py                 # Training script with checkpointing
└── training_328.sh              # Training automation script
```

## Documentation

For detailed information about the system:

- **[Architecture Overview](docs/architecture.md)**: High-level system design and components
- **[Model Details](docs/model-details.md)**: Deep dive into the U-Net architecture and training
- **[Data Flow](docs/data-flow.md)**: How data flows through training and inference pipelines
- **[Training Guide](docs/training-guide.md)**: Complete guide to training your own models
- **[API Reference](docs/api-reference.md)**: REST API endpoints and usage examples

### Visual Diagrams

The documentation includes detailed PlantUML diagrams that illustrate:
- System architecture and component relationships
- U-Net neural network structure
- Training and inference data flows
- Audio processing pipelines
- API architecture

View the diagrams at [PlantUML Online](http://www.plantuml.com/plantuml/uml/) by copying the code from `docs/diagrams/` or install a PlantUML viewer in your IDE.

## Best Practices Implemented

- **Type Hints**: Extensive use of Python type annotations
- **Error Handling**: Graceful fallbacks and detailed error logging
- **Modularity**: Clear separation of concerns between components
- **Documentation**: Comprehensive docstrings and comments
- **Configuration**: Centralized, type-safe configuration with Pydantic
- **Asynchronous Processing**: Efficient streaming with FastAPI async endpoints
- **Dependency Injection**: Services use DI pattern for better testability
- **Code Reuse**: Inheritance and composition for shared functionality

## Acknowledgements
This code is based on [Ultralight-Digital-Human](https://github.com/anliyuan/Ultralight-Digital-Human) and [SyncTalk](https://github.com/ZiqiaoPeng/SyncTalk). We thank the authors for their excellent work.
