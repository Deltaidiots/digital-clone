"""
Models service for SyncTalk_2D.

This module manages the loading and initialization of deep learning models used
for talking head animation. It follows the Singleton pattern to ensure models are
loaded only once and shared across the application.

Classes:
    ModelService: Service for loading and providing access to neural network models

Dependencies:
    - PyTorch for model loading and inference
    - OpenCV for image processing
    - Custom models (UNet, AudioEncoder) for animation generation
"""

import torch
import cv2
import numpy as np
import logging
from typing import Tuple, Dict, Any, Optional
from unet_328 import Model
from utils import AudioEncoder
from config import settings

class ModelService:
    """
    Service for loading and managing deep learning models used in SyncTalk_2D.
    
    This service handles loading the U-Net model for face generation, the audio
    encoder for processing speech, and the template image used as the animation base.
    It follows the Single Responsibility Principle by focusing exclusively on model
    management.
    
    Attributes:
        device: The PyTorch device (CPU or GPU) used for computation
        model_name: Name of the currently loaded model
        net: The main U-Net generator model
        audio_encoder: The audio encoder model for processing speech features
        img: The template image used as the base for animation
        lms: Facial landmarks for the template image
        crop_coords: Pre-calculated face crop coordinates (xmin, ymin, xmax, ymax)
    """
    
    def __init__(self, config=settings.models):
        """
        Initialize the model service by loading all required models and assets.
        
        Args:
            config: Configuration object with model paths and settings
        
        Raises:
            RuntimeError: If model loading fails
            FileNotFoundError: If model files or template image are not found
        """
        self.device = torch.device(config.device)
        logging.info(f"Loading models onto device: {self.device}")

        # Extract model name from checkpoint path (e.g., "./checkpoint/May/4.pth" → "May")
        self.model_name = config.checkpoint_path.split('/')[-2] if '/' in config.checkpoint_path else "Unknown"
        logging.info(f"Using model: {self.model_name}")
        
        # Load Generator
        try:
            self.net = Model(6, mode="ave").to(self.device)
            self.net.load_state_dict(torch.load(config.checkpoint_path, map_location=self.device))
            self.net.eval()
            logging.info(f"Successfully loaded generator model from {config.checkpoint_path}")
        except Exception as e:
            logging.error(f"Failed to load generator model: {str(e)}", exc_info=True)
            raise RuntimeError(f"Failed to load generator model: {str(e)}") from e

        # Load Audio Encoder
        try:
            self.audio_encoder = AudioEncoder().to(self.device).eval()
            ckpt = torch.load(config.audio_encoder_ckpt, map_location=self.device)
            self.audio_encoder.load_state_dict({f'audio_encoder.{k}': v for k, v in ckpt.items()})
            logging.info(f"Successfully loaded audio encoder from {config.audio_encoder_ckpt}")
        except Exception as e:
            logging.error(f"Failed to load audio encoder: {str(e)}", exc_info=True)
            raise RuntimeError(f"Failed to load audio encoder: {str(e)}") from e

        # Load Template Image and Landmarks
        try:
            self.img = cv2.imread(config.template_img_path)
            if self.img is None:
                raise FileNotFoundError(f"Failed to load template image: {config.template_img_path}")
                
            lms_list = []
            with open(config.template_lms_path) as f:
                for line in f.read().splitlines():
                    lms_list.append(np.array(line.split(" "), dtype=np.float32))
            self.lms = np.array(lms_list, dtype=np.int32)
            
            logging.info(f"Successfully loaded template image and landmarks")
        except Exception as e:
            logging.error(f"Failed to load template image or landmarks: {str(e)}", exc_info=True)
            raise RuntimeError(f"Failed to load template assets: {str(e)}") from e
        
        # Pre-calculate crop coordinates
        self.crop_coords = self._calculate_crop_coords()
        logging.info("ModelService initialized successfully.")
    
    def _calculate_crop_coords(self) -> Tuple[int, int, int, int]:
        """
        Calculate face crop coordinates from landmarks.
        
        Returns:
            Tuple[int, int, int, int]: (xmin, ymin, xmax, ymax) crop coordinates
        """
        xmin, ymin = self.lms[1][0], self.lms[52][1]
        xmax = self.lms[31][0]
        width = xmax - xmin
        ymax = ymin + width
        return (xmin, ymin, xmax, ymax)

# Singleton instance to be used across the app
model_service = ModelService()
