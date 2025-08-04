"""
TTS Service - Uses the Piper TTS system to convert text to speech.

This module provides a simple, reliable interface to the Piper TTS system.
It includes fallback mechanisms for handling missing espeak data directories
and other common issues that might occur during initialization.

The module supports multiple TTS service implementations through the TTSService
abstract base class, with the primary implementation being PiperTTSService.
"""
from abc import ABC, abstractmethod
import os
import json
import pathlib
import numpy as np
import logging
from typing import Optional, Dict, Any, Iterator, List
from config import settings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Import piper after logging is configured
try:
    from piper import PiperVoice
    from piper.config import SynthesisConfig
    PIPER_AVAILABLE = True
except ImportError:
    logging.error("Failed to import piper. Make sure it's installed with 'pip install piper-tts'")
    PIPER_AVAILABLE = False

class TTSService(ABC):
    """Abstract base class for text-to-speech services."""
    
    @abstractmethod
    def synthesize(self, text: str) -> bytes:
        """
        Convert text to audio waveform bytes.
        
        Args:
            text: The input text to synthesize
            
        Returns:
            bytes: Raw waveform bytes (16-bit PCM)
        """
        pass

    @property
    @abstractmethod
    def sample_rate(self) -> int:
        """Get the sample rate of the synthesized audio."""
        pass
    
    @abstractmethod
    def get_audio_duration(self, audio_data: bytes) -> float:
        """
        Calculate the duration in seconds of the audio data.
        
        Args:
            audio_data: The raw audio data bytes
            
        Returns:
            float: Duration in seconds
        """
        pass
    
class PiperTTSService(TTSService):
    """
    TTS Service implementation using Piper.
    
    This service uses the Piper TTS library to convert text to speech.
    It implements a simplified approach that uses PiperVoice.load() for a more reliable setup.
    
    The service handles multiple configurations:
    - Properly manages espeak_data_dir parameter (required by Piper)
    - Falls back gracefully when required paths don't exist
    - Provides detailed logging for troubleshooting
    - Implements error handling for common TTS issues
    
    The service initializes with settings from config.py but can be customized
    by passing a different configuration object to the constructor.
    """
    def __init__(self, config=settings.tts):
        """
        Initialize the Piper TTS service.
        
        Args:
            config: Configuration object with model_path and model_config_path
        """
        if not PIPER_AVAILABLE:
            raise ImportError("Piper TTS not available. Please install with 'pip install piper-tts'.")
            
        self.model_path = config.model_path
        self.model_config_path = config.model_config_path
        self._sample_rate = 16000  # Default, will be updated when voice is loaded
        self.voice = None
        
        logging.info(f"Initializing PiperTTSService with model: {self.model_path} and config: {self.model_config_path}")
        
        # Check if model file exists
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model file not found: {self.model_path}")
            
        # Set up the espeak_data_dir
        espeak_data_dir = config.espeak_data_dir if hasattr(config, 'espeak_data_dir') else config.espeak_data_path
        
        # Check if espeak_data_dir exists and handle accordingly
        if espeak_data_dir and not os.path.exists(espeak_data_dir):
            logging.warning(f"Espeak data path {espeak_data_dir} not found, using environment variable")
            # Try to use environment variable
            espeak_data_dir = os.environ.get("ESPEAK_DATA_PATH")
        
        # Handle case where espeak_data_dir is None or empty
        if not espeak_data_dir:
            logging.warning("No espeak_data_dir provided, Piper will use default locations")
            # Load the voice without the espeak_data_dir parameter
            try:
                self.voice = PiperVoice.load(
                    self.model_path,
                    config_path=self.model_config_path,
                    use_cuda=False  # Use CPU for better compatibility
                )
            except Exception as e:
                logging.error(f"Failed to load Piper voice: {e}", exc_info=True)
                raise
        else:
            # Load the voice with the espeak_data_dir parameter
            try:
                self.voice = PiperVoice.load(
                    self.model_path,
                    config_path=self.model_config_path,
                    espeak_data_dir=espeak_data_dir,
                    use_cuda=False  # Use CPU for better compatibility
                )
                self._sample_rate = self.voice.config.sample_rate
                logging.info(f"Successfully loaded Piper voice with sample rate {self._sample_rate}")
            except Exception as e:
                logging.error(f"Failed to load Piper voice: {e}", exc_info=True)
                raise

    @property
    def sample_rate(self) -> int:
        """Get the sample rate of the synthesized audio."""
        return self._sample_rate

    def synthesize(self, text: str) -> bytes:
        """
        Synthesize text to speech using Piper and return the raw waveform bytes.
        
        Args:
            text: Input text to synthesize
            
        Returns:
            bytes: Raw audio data as 16-bit PCM
        """
        if not text or not text.strip():
            logging.warning("Empty text provided to synthesize")
            return b''
        
        if not self.voice:
            logging.error("TTS voice not initialized")
            return b''
            
        try:
            logging.info(f"Synthesizing text: '{text}'")
            
            # Use BytesIO to capture WAV data in memory
            import io
            import wave
            
            wav_bytes = io.BytesIO()
            with wave.open(wav_bytes, "wb") as wav_file:
                # This simpler approach handles all the internal complexity
                self.voice.synthesize_wav(text, wav_file)
                
            # Get the raw bytes (excluding WAV header)
            wav_bytes.seek(0)
            with wave.open(wav_bytes, "rb") as wav_file:
                n_frames = wav_file.getnframes()
                audio_bytes = wav_file.readframes(n_frames)
            
            logging.info(f"Generated audio of length {len(audio_bytes)} bytes")
            return audio_bytes
            
        except Exception as e:
            logging.error(f"Piper synthesis error: {str(e)}", exc_info=True)
            return b''

    def get_audio_duration(self, audio_data: bytes) -> float:
        """
        Calculate the duration in seconds of the audio data.
        
        Args:
            audio_data: The raw audio data bytes (16-bit PCM)
            
        Returns:
            float: Duration in seconds
        """
        if not audio_data:
            return 0.0
            
        # For 16-bit PCM audio, we need 2 bytes per sample
        num_samples = len(audio_data) // 2
        duration = num_samples / self._sample_rate
        return duration

# Singleton instance creation function
def create_tts_service() -> TTSService:
    """Create and return a TTSService instance."""
    try:
        service = PiperTTSService()
        logging.info("Successfully initialized TTSService")
        return service
    except Exception as e:
        logging.critical(f"Failed to initialize TTSService: {e}", exc_info=True)
        # Create a dummy TTS service for graceful degradation
        class DummyTTSService(TTSService):
            @property
            def sample_rate(self) -> int:
                return 16000
                
            def synthesize(self, text: str) -> bytes:
                logging.error("Using dummy TTS service - no audio will be generated")
                return b''
                
            def get_audio_duration(self, audio_data: bytes) -> float:
                return 0.0
                
        logging.warning("Using DummyTTSService as fallback")
        return DummyTTSService()

# Singleton instance
tts_service = create_tts_service()
