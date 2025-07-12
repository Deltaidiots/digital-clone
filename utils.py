import librosa
import librosa.filters
from scipy import signal
from os.path import basename
import numpy as np
import logging
from typing import Union, List, Dict, Tuple, Optional, Any, Callable

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F




class Conv2d(nn.Module):
    """
    Custom 2D convolutional layer with optional residual connections.
    
    This layer combines convolution, batch normalization, and activation,
    with support for residual connections for deeper networks.
    
    Attributes:
        conv_block: Sequential block of Conv2d and BatchNorm2d
        act: Activation function (ReLU or LeakyReLU)
        residual: Whether to use residual connections
    """
    
    def __init__(
        self, 
        cin: int, 
        cout: int, 
        kernel_size: Union[int, Tuple[int, int]], 
        stride: Union[int, Tuple[int, int]], 
        padding: Union[int, Tuple[int, int]], 
        residual: bool = False, 
        leakyReLU: bool = False, 
        *args, 
        **kwargs
    ) -> None:
        """
        Initialize the Conv2d layer.
        
        Args:
            cin: Number of input channels
            cout: Number of output channels
            kernel_size: Size of the convolving kernel
            stride: Stride of the convolution
            padding: Padding added to all sides of the input
            residual: Whether to use residual connections
            leakyReLU: Whether to use LeakyReLU instead of ReLU
            *args: Additional arguments for the parent class
            **kwargs: Additional keyword arguments for the parent class
        """
        super().__init__(*args, **kwargs)
        self.conv_block = nn.Sequential(
            nn.Conv2d(cin, cout, kernel_size, stride, padding),
            nn.BatchNorm2d(cout)
        )
        if leakyReLU:
            self.act = nn.LeakyReLU(0.02)
        else:
            self.act = nn.ReLU()
        self.residual = residual

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the Conv2d layer.
        
        Args:
            x: Input tensor
            
        Returns:
            torch.Tensor: Output tensor after convolution, batch norm, and activation
        """
        out = self.conv_block(x)
        if self.residual:
            out += x
        return self.act(out)


class AudioEncoder(nn.Module):
    """
    Audio encoder module for extracting features from mel spectrograms.
    
    This module processes mel spectrograms through a series of convolutional layers
    to extract audio features. It expects a 4D input tensor with shape:
    [batch_size, channels, features, time]
    
    The output is a flattened feature vector with 512 dimensions per batch item.
    
    Attributes:
        audio_encoder: Sequential network of convolutional layers
    """
    def __init__(self) -> None:
        """
        Initialize the AudioEncoder module with a series of Conv2d layers.
        """
        super(AudioEncoder, self).__init__()

        self.audio_encoder = nn.Sequential(
            Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            Conv2d(32, 32, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(32, 32, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(32, 64, kernel_size=3, stride=(3, 1), padding=1),
            Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(64, 128, kernel_size=3, stride=3, padding=1),
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(128, 256, kernel_size=3, stride=(3, 2), padding=1),
            Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(256, 512, kernel_size=3, stride=1, padding=0),
            Conv2d(512, 512, kernel_size=1, stride=1, padding=0), )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the AudioEncoder.
        
        Args:
            x: Input tensor with shape [batch_size, channels, features, time]
            
        Returns:
            torch.Tensor: Output feature tensor with shape [batch_size, 512]
        """
        out = self.audio_encoder(x)
        out = out.squeeze(2).squeeze(2)

        return out
    
def get_audio_features(features: np.ndarray, index: int) -> torch.Tensor:
    """
    Extract a window of audio features centered around a given index.
    
    Args:
        features: Audio features matrix
        index: Center index for the feature window
        
    Returns:
        torch.Tensor: Window of audio features with padding if needed
    """
    left = index - 8
    right = index + 8
    pad_left = 0
    pad_right = 0
    if left < 0:
        pad_left = -left
        left = 0
    if right > features.shape[0]:
        pad_right = right - features.shape[0]
        right = features.shape[0]
    auds = torch.from_numpy(features[left:right])
    if pad_left > 0:
        auds = torch.cat([torch.zeros_like(auds[:pad_left]), auds], dim=0)
    if pad_right > 0:
        auds = torch.cat([auds, torch.zeros_like(auds[:pad_right])], dim=0) # [8, 16]
    return auds


def load_wav(path: str, sr: int) -> np.ndarray:
    """
    Load a WAV file with the specified sample rate.
    
    Args:
        path: Path to the WAV file
        sr: Target sample rate
        
    Returns:
        np.ndarray: Audio waveform
    """
    return librosa.core.load(path, sr=sr)[0]


def preemphasis(wav: np.ndarray, k: float) -> np.ndarray:
    """
    Apply pre-emphasis filter to the audio waveform.
    
    Args:
        wav: Audio waveform
        k: Pre-emphasis coefficient
        
    Returns:
        np.ndarray: Filtered audio waveform
    """
    return signal.lfilter([1, -k], [1], wav)


def melspectrogram(wav: np.ndarray) -> np.ndarray:
    """
    Convert an audio waveform to a mel spectrogram.
    
    Args:
        wav: Audio waveform
        
    Returns:
        np.ndarray: Mel spectrogram
    """
    D = _stft(preemphasis(wav, 0.97))
    S = _amp_to_db(_linear_to_mel(np.abs(D))) - 20

    return _normalize(S)


def _stft(y: np.ndarray) -> np.ndarray:
    """
    Apply Short-Time Fourier Transform to the audio waveform.
    
    Args:
        y: Audio waveform
        
    Returns:
        np.ndarray: STFT result
    """
    return librosa.stft(y=y, n_fft=800, hop_length=200, win_length=800)


def _linear_to_mel(spectogram: np.ndarray) -> np.ndarray:
    """
    Convert linear spectrogram to mel spectrogram.
    
    Args:
        spectogram: Linear spectrogram
        
    Returns:
        np.ndarray: Mel spectrogram
    """
    global _mel_basis
    _mel_basis = _build_mel_basis()
    return np.dot(_mel_basis, spectogram)


def _build_mel_basis() -> np.ndarray:
    """
    Build a mel filter bank.
    
    Returns:
        np.ndarray: Mel filter bank matrix
    """
    return librosa.filters.mel(sr=16000, n_fft=800, n_mels=80, fmin=55, fmax=7600)


def _amp_to_db(x: np.ndarray) -> np.ndarray:
    """
    Convert amplitude to decibels.
    
    Args:
        x: Input amplitude
        
    Returns:
        np.ndarray: Amplitude in decibels
    """
    min_level = np.exp(-5 * np.log(10))
    return 20 * np.log10(np.maximum(min_level, x))


def _normalize(S: np.ndarray) -> np.ndarray:
    """
    Normalize the spectrogram to a specific range.
    
    Args:
        S: Input spectrogram
        
    Returns:
        np.ndarray: Normalized spectrogram
    """
    return np.clip((2 * 4.) * ((S - -100) / (--100)) - 4., -4., 4.)


def get_audio_features_realtime(
    waveform: Union[np.ndarray, torch.Tensor],
    audio_encoder: nn.Module,
    device: Union[str, torch.device],
    mode: str = "ave"
) -> torch.Tensor:
    """
    Extracts audio features in real-time from audio waveform data.
    
    This function processes audio data into mel spectrograms and extracts features
    using the provided audio encoder. It handles proper tensor dimensionality
    to ensure compatibility with Conv2D layers.
    
    Args:
        waveform: Audio waveform data as numpy array or torch tensor
        audio_encoder: The audio encoder model
        device: The device to run inference on (CPU or CUDA)
        mode: Audio encoding mode, affects the output reshape dimensions
            "ave" - Reshape to [1, 32, 16, 16] for U-Net model
            "hubert" - Reshape to [1, 32, 32, 32] for other models
    
    Returns:
        Extracted audio features, reshaped according to the mode parameter
        
    Raises:
        RuntimeError: If audio preprocessing or feature extraction fails
    """
    # Convert to tensor if numpy array
    if isinstance(waveform, np.ndarray):
        wave_tensor = torch.FloatTensor(waveform)
    else:
        wave_tensor = waveform
        
    # Move to device
    wave_tensor = wave_tensor.to(device)
    
    try:
        # Process audio into mel spectrogram
        # Assuming sample rate of 16000 for audio processing
        mel_tensor = audio_preprocessing(wave_tensor.squeeze(0))
        mel_tensor = mel_tensor.to(device)
    except Exception as e:
        # Log the error and raise with more context
        logging.error(f"Error in audio preprocessing: {str(e)}", exc_info=True)
        raise RuntimeError(f"Failed to process audio: {str(e)}") from e
    
    # The audio encoder expects a window of 16 mels
    if mel_tensor.shape[-1] < 16:
        # Pad if not enough data, though in a streaming setup this should be handled by buffer size
        mel_tensor = torch.nn.functional.pad(mel_tensor, (0, 16 - mel_tensor.shape[-1]), 'constant', 0)
    
    mel_chunk = mel_tensor[..., :16]
    
    # Add channel dimension if needed
    if mel_chunk.dim() == 3:  # [batch, features, time]
        # Add channel dimension for Conv2d: [batch, channels, features, time]
        mel_chunk = mel_chunk.unsqueeze(1)

    with torch.no_grad():
        audio_feature = audio_encoder(mel_chunk) # (1, 512)

    # Reshape to match model input
    """
    Audio Feature Reshaping Process:
    
    1. The audio encoder outputs a 1D tensor of size 512
    2. This needs to be reshaped to match the expected input of the AudioConv models
    3. For AVE mode:
       - First reshape to [batch, 32, 4, 4] (32 channels, 4x4 spatial dims)
       - Then upsample to [batch, 32, 16, 16] using bilinear interpolation
    4. For Hubert mode:
       - First reshape to [batch, 32, 4, 4]
       - Then upsample to [batch, 32, 32, 32]
    """
    # Reshape to match model input
    if mode == "ave":
        # For AVE mode, expand 512 features to [batch, 32, 4, 4] and then upsample to [batch, 32, 16, 16]
        audio_feature = audio_feature.reshape(1, 32, 4, 4)  # Explicitly set batch size to 1
        # Upsample to 16x16 spatial dimensions
        audio_feature = torch.nn.functional.interpolate(
            audio_feature, 
            size=(16, 16), 
            mode='bilinear', 
            align_corners=False
        )
    elif mode == "hubert":
        audio_feature = audio_feature.reshape(1, 32, 4, 4)  # Explicitly set batch size to 1
        # Upsample to 32x32 spatial dimensions
        audio_feature = torch.nn.functional.interpolate(
            audio_feature, 
            size=(32, 32), 
            mode='bilinear', 
            align_corners=False
        )
    # Add other modes if necessary

    # Alternative approach (commented out) - zero pad to reach the required dimensions
    """
    # Alternative approach using zero padding:
    if mode == "ave":
        # Create a tensor of zeros with the target shape
        reshaped_feature = torch.zeros(1, 32, 16, 16, device=audio_feature.device)
        # Fill in as much as we can from the original feature
        # This preserves the original 512 values while padding the rest with zeros
        flat_feature = audio_feature.reshape(-1)
        flat_reshaped = reshaped_feature.reshape(-1)
        flat_reshaped[:flat_feature.shape[0]] = flat_feature
        audio_feature = reshaped_feature
    """

    return audio_feature


class AudDataset(object):
    """
    Dataset for loading and processing audio files for model training or inference.
    
    This class handles the extraction of mel spectrograms from audio files and
    provides methods to crop audio windows for frame-by-frame processing.
    
    Attributes:
        orig_mel: Original mel spectrogram
        data_len: Length of the dataset in frames
    """
    
    def __init__(self, wavpath: str) -> None:
        """
        Initialize the audio dataset.
        
        Args:
            wavpath: Path to the WAV file
        """
        wav = load_wav(wavpath, 16000)

        self.orig_mel = melspectrogram(wav).T
        self.data_len = int((self.orig_mel.shape[0] - 16) / 80. * float(25)) + 2

    def get_frame_id(self, frame: str) -> int:
        """
        Get the frame ID from a frame filename.
        
        Args:
            frame: Frame filename
            
        Returns:
            int: Frame ID
        """
        return int(basename(frame).split('.')[0])

    def crop_audio_window(self, spec: np.ndarray, start_frame: Union[int, str]) -> np.ndarray:
        """
        Crop an audio window from the spectrogram.
        
        Args:
            spec: Spectrogram to crop from
            start_frame: Starting frame (either frame ID or frame filename)
            
        Returns:
            np.ndarray: Cropped audio window
        """
        if isinstance(start_frame, int):
            start_frame_num = start_frame
        else:
            start_frame_num = self.get_frame_id(start_frame)
        start_idx = int(80. * (start_frame_num / float(25)))

        end_idx = start_idx + 16
        if end_idx > spec.shape[0]:
            # Adjust window if it exceeds the spectrogram boundary
            end_idx = spec.shape[0]
            start_idx = end_idx - 16

        return spec[start_idx: end_idx, :]

    def __len__(self) -> int:
        """
        Get the length of the dataset.
        
        Returns:
            int: Number of frames in the dataset
        """
        return self.data_len

    def __getitem__(self, idx: int) -> torch.Tensor:
        """
        Get a mel spectrogram for a specific frame index.
        
        Args:
            idx: Frame index
            
        Returns:
            torch.Tensor: Mel spectrogram tensor
            
        Raises:
            Exception: If the mel spectrogram shape is not as expected
        """
        mel = self.crop_audio_window(self.orig_mel.copy(), idx)
        if (mel.shape[0] != 16):
            raise Exception('mel.shape[0] != 16')
        mel = torch.FloatTensor(mel.T).unsqueeze(0)

        return mel


def audio_preprocessing(wav_tensor: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
    """
    Convert audio waveform tensor to mel spectrogram tensor for real-time processing.
    
    This function handles the conversion of raw audio waveforms to mel spectrograms,
    which are needed as input for the audio encoder model.
    
    Args:
        wav_tensor: Audio waveform tensor or numpy array
        
    Returns:
        Mel spectrogram tensor with proper dimensions for model input
        
    Raises:
        ValueError: If audio conversion fails
    """
    # Convert tensor to numpy for processing with librosa
    if torch.is_tensor(wav_tensor):
        wav_numpy = wav_tensor.cpu().numpy()
    else:
        wav_numpy = wav_tensor
        
    # Use the existing melspectrogram function
    mel_spec = melspectrogram(wav_numpy)
    
    # Convert back to tensor
    mel_tensor = torch.FloatTensor(mel_spec)
    
    # Add batch dimension if needed
    if mel_tensor.dim() == 2:
        mel_tensor = mel_tensor.unsqueeze(0)
        
    return mel_tensor
