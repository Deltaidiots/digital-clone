"""
Configuration module for SyncTalk_2D application.

This module provides a centralized configuration system for the application using
Pydantic models for type safety and validation. The configuration is organized into
logical sections (models, TTS, debugging) with reasonable defaults.

Classes:
    ModelConfig: Configuration for neural network models and device settings
    TTSConfig: Configuration for Text-to-Speech services
    DebugConfig: Configuration for debugging and diagnostics
    AppConfig: Root configuration container that combines all config sections

Usage:
    from config import settings
    
    # Access configuration values
    device = settings.models.device
    tts_model_path = settings.tts.model_path
    is_debug_enabled = settings.debug.enabled
"""

from pydantic import BaseModel, ConfigDict, Field
import torch
from typing import Optional

class ModelConfig(BaseModel):
    """
    Configuration for neural network models and device settings.
    
    Attributes:
        device: The device to use for computation (cuda or cpu)
        checkpoint_path: Path to the trained model checkpoint
        audio_encoder_ckpt: Path to the audio encoder checkpoint
        template_img_path: Path to the template image for animation
        template_lms_path: Path to the landmarks file for the template image
    """
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    checkpoint_path: str = "./checkpoint/Awais/5.pth" 
    audio_encoder_ckpt: str = 'model/checkpoints/audio_visual_encoder.pth'
    template_img_path: str = "./dataset/Awais/full_body_img/0.jpg"
    template_lms_path: str = "./dataset/Awais/landmarks/0.lms"

class TTSConfig(BaseModel):
    """
    Configuration for Text-to-Speech services.
    
    Attributes:
        model_path: Path to the TTS model file
        model_config_path: Path to the TTS model configuration
        sample_rate: Audio sample rate for TTS output
        espeak_data_dir: Directory containing eSpeak data files for phoneme processing
        espeak_data_path: Alias for espeak_data_dir (kept for backward compatibility)
    """
    model_config = ConfigDict(protected_namespaces=())
    model_path: str = "./models/en_US-lessac-low.onnx"
    model_config_path: str = "./models/en_US-lessac-low.onnx.json"
    sample_rate: int = 16000
    espeak_data_dir: str = "./local_espeak_data" 
    espeak_data_path: str = "./local_espeak_data"  # Kept for backward compatibility

class DebugConfig(BaseModel):
    """
    Configuration for debugging and diagnostic features.
    
    Attributes:
        enabled: Whether debugging features are enabled
        save_frames: Whether to save generated frames to disk
        save_audio: Whether to save generated audio to disk
        frames_dir: Directory to save debug frames
        audio_dir: Directory to save debug audio
    """
    enabled: bool = False
    save_frames: bool = False
    save_audio: bool = False
    frames_dir: str = "./debug/frames"
    audio_dir: str = "./debug/audio"

# Face enhancement config removed

class AppConfig(BaseModel):
    """
    Root configuration container combining all configuration sections.
    
    Attributes:
        models: Configuration for models and device settings
        tts: Configuration for Text-to-Speech services
        debug: Configuration for debugging and diagnostics
    """
    models: ModelConfig = Field(default_factory=ModelConfig)
    tts: TTSConfig = Field(default_factory=TTSConfig)
    debug: DebugConfig = Field(default_factory=DebugConfig)

# Create a single, immutable config instance
settings = AppConfig()
