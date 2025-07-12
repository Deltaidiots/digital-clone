import numpy as np
import torch
import cv2
import asyncio
import time
import io
import logging
import os
import datetime
from typing import Tuple, Generator, Optional, Any, AsyncGenerator, Dict, Union, List, Callable
from services.models import ModelService
from services.tts import TTSService
from utils import get_audio_features_realtime, get_audio_features, melspectrogram
from config import settings

class FrameGenerationService:
    """
    Service for generating animated talking head frames from text or audio input.
    
    This service handles the process of converting text to speech or processing audio files,
    extracting audio features, and generating animated frames using a deep learning model.
    
    Attributes:
        models: The model service providing neural networks and the template image
        tts: The text-to-speech service for synthesizing speech
        _cached_preprocess: Cached preprocessed image data to avoid redundant computation
    """
    
    def __init__(self, model_service: ModelService, tts_service: TTSService) -> None:
        """
        Initialize the frame generation service.
        
        Args:
            model_service: Service that provides models and template image
            tts_service: Service for text-to-speech synthesis
        """
        self.models = model_service
        self.tts = tts_service
        # Cache the preprocessed image data
        self._cached_preprocess: Optional[Tuple[torch.Tensor, Tuple[int, int], np.ndarray]] = None
        
        # Create debug directories if needed
        if settings.debug.enabled:
            self._setup_debug_directories()
    
    def _setup_debug_directories(self) -> None:
        """
        Create directories for debug files if they don't exist.
        """
        if settings.debug.save_frames and not os.path.exists(settings.debug.frames_dir):
            os.makedirs(settings.debug.frames_dir, exist_ok=True)
            logging.info(f"Created debug frames directory: {settings.debug.frames_dir}")
            
        if settings.debug.save_audio and not os.path.exists(settings.debug.audio_dir):
            os.makedirs(settings.debug.audio_dir, exist_ok=True)
            logging.info(f"Created debug audio directory: {settings.debug.audio_dir}")
    
    def _save_debug_frame(self, frame: np.ndarray, session_id: str, frame_index: int) -> None:
        """
        Save a single frame to the debug directory if debug mode is enabled.
        
        Args:
            frame: The frame image to save
            session_id: Unique identifier for this generation session
            frame_index: The sequential index of this frame
        """
        if not settings.debug.enabled or not settings.debug.save_frames:
            return
            
        frame_path = os.path.join(
            settings.debug.frames_dir,
            f"{session_id}_{frame_index:04d}.jpg"
        )
        try:
            cv2.imwrite(frame_path, frame)
        except Exception as e:
            logging.error(f"Failed to save debug frame: {str(e)}")
    
    def _save_debug_audio(self, audio_data: bytes, session_id: str, is_tts: bool = True) -> None:
        """
        Save audio data to the debug directory if debug mode is enabled.
        
        Args:
            audio_data: The audio data to save
            session_id: Unique identifier for this generation session
            is_tts: Whether this is TTS-generated audio (True) or uploaded audio (False)
        """
        if not settings.debug.enabled or not settings.debug.save_audio:
            return
            
        prefix = "tts" if is_tts else "upload"
        audio_path = os.path.join(
            settings.debug.audio_dir,
            f"{prefix}_{session_id}.wav"
        )
        try:
            with open(audio_path, "wb") as f:
                f.write(audio_data)
            logging.info(f"Saved debug audio to {audio_path}")
        except Exception as e:
            logging.error(f"Failed to save debug audio: {str(e)}")
    
    def _preprocess_image(self) -> Tuple[torch.Tensor, Tuple[int, int], np.ndarray]:
        """
        Preprocess the template image for the model.
        
        This function prepares the template image by cropping, resizing, and masking it
        to create the proper input for the neural network model. The results are cached
        for efficiency in subsequent calls.
        
        Returns:
            tuple: A tuple containing:
                - model_input (torch.Tensor): The prepared tensor input for the model
                - original_dimensions (Tuple[int, int]): The height and width of the original crop
                - resized_image (np.ndarray): The resized image at 328x328 resolution
        """
        # Use cached preprocessing if available
        if self._cached_preprocess is not None:
            return self._cached_preprocess
            
        # Get crop coordinates from the landmark detection
        xmin, ymin, xmax, ymax = self.models.crop_coords
        crop_img = self.models.img[ymin:ymax, xmin:xmax]
        h, w = crop_img.shape[:2]
        
        # Resize to model input size
        crop_img_r = cv2.resize(crop_img, (328, 328), interpolation=cv2.INTER_CUBIC)
        
        # Create a masked version of the image (for the model input)
        img_masked = crop_img_r[4:324, 4:324].copy()
        img_masked = cv2.rectangle(img_masked, (5, 5, 310, 305), (0, 0, 0), -1)
        
        # Convert to PyTorch tensors
        img_masked_T = torch.from_numpy((img_masked.transpose(2,0,1) / 255.0).astype(np.float32))
        img_real_ex_T = torch.from_numpy((crop_img_r[4:324, 4:324].transpose(2,0,1) / 255.0).astype(np.float32))
        
        # Combine into a single input tensor
        model_input = torch.cat([img_real_ex_T, img_masked_T], axis=0)[None].to(self.models.device)
        
        # Cache the results
        self._cached_preprocess = (model_input, (h, w), crop_img_r)
        return self._cached_preprocess

    async def generate_static_frame(self) -> bytes:
        """
        Generate a static frame (no animation) to show when the app starts.
        
        This provides the initial frame that is displayed before any animation begins.
        It's essentially the template image prepared for HTTP streaming.
        
        Returns:
            bytes: JPEG-encoded frame with HTTP multipart content headers
        """
        # Get the template image at its original dimensions
        img = self.models.img.copy()
            
        # Encode as high-quality JPEG
        jpeg = self._encode_high_quality_jpeg(img)
        frame_bytes = jpeg.tobytes()
        
        # Format for multipart HTTP response
        return b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n'

    async def generate_frames(self, text: str, audio_duration: float = 0.0) -> AsyncGenerator[bytes, None]:
        """
        Generate animated frames from text input.
        
        This method converts text to speech using the TTS service, then processes the
        audio to extract features and generate animated frames showing the template face
        speaking the provided text.
        
        Args:
            text: The text to synthesize and animate
            audio_duration: Optional known audio duration in seconds for better sync
            
        Yields:
            bytes: A series of JPEG-encoded frames with HTTP multipart framing
            
        Raises:
            RuntimeError: If text-to-speech synthesis fails or audio processing fails
        """
        # Create a unique session ID for this generation
        session_id = f"text_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
        logging.info(f"Starting frame generation for text input (session: {session_id})")
        
        # First send a static frame
        yield await self.generate_static_frame()
        
        try:
            # Synthesize speech
            waveform_bytes = self.tts.synthesize(text)
            if not waveform_bytes:  # Handle empty result
                logging.error("TTS synthesis returned empty result")
                return
                
            # Save the audio for debugging if enabled
            if settings.debug.enabled and settings.debug.save_audio:
                self._save_debug_audio(waveform_bytes, session_id, is_tts=True)
                
            # Convert to numpy array
            waveform_np = np.frombuffer(waveform_bytes, dtype=np.int16).astype(np.float32) / 32767.0

            # Get preprocessed image data
            img_concat_T, (h, w), crop_img_r = self._preprocess_image()

            # Convert to mel spectrogram
            mel = melspectrogram(waveform_np).T
            mel_tensor = torch.FloatTensor(mel)
            
            # Process in smaller chunks to enable real-time streaming
            # Instead of processing all audio at once, we'll extract features in smaller windows
            model = self.models.audio_encoder
            device = self.models.device
            
            # Calculate an adjusted frame rate based on audio duration
            fps_target = self._calculate_adjusted_fps(mel_tensor, audio_duration)
            frame_time = 1.0 / fps_target
            last_frame_time = time.time()
            
            # We process in smaller chunks (e.g., 50 frames at a time)
            # This allows frames to start streaming while features are still being extracted
            chunk_size = 50
            step_size = 4  # Step by 4 frames for ~25 fps
            context_size = 16  # Each audio window is 16 frames
            
            # Calculate number of frames
            num_frames = max(1, (mel_tensor.shape[0] - context_size) // step_size + 1)
            
            # Process and stream audio features in chunks
            frame_idx = 0
            audio_features_cache = []  # Cache for already processed features
            
            logging.info(f"Processing {num_frames} frames for text input (session: {session_id})")
            
            # Process the first chunk of audio features
            chunk_end = min(chunk_size, num_frames)
            for i in range(0, chunk_end):
                mel_idx = i * step_size
                if mel_idx + context_size > mel_tensor.shape[0]:
                    mel_idx = mel_tensor.shape[0] - context_size
                
                mel_window = mel_tensor[mel_idx:mel_idx+context_size].T.unsqueeze(0)  # [1, 80, 16]
                mel_window = mel_window.unsqueeze(1).to(device)  # [1, 1, 80, 16]
                
                with torch.no_grad():
                    audio_feature = model(mel_window)  # [1, 512]
                    audio_features_cache.append(audio_feature.cpu().numpy())
            
            # We need at least 8 frames of context for proper get_audio_features
            # Add padding at the beginning for smooth transitions
            if audio_features_cache:
                first_frame = audio_features_cache[0]
                audio_features_cache = [first_frame] * 8 + audio_features_cache
            
            # Now, process frames chunk by chunk
            max_retries = 10  # Maximum number of retries to prevent infinite loops
            retry_count = 0
            
            while frame_idx < num_frames:
                # Process more audio features if needed
                next_chunk_start = len(audio_features_cache) - 8  # Account for padding
                need_more_features = next_chunk_start + 8 < min(frame_idx + 10, num_frames)
                
                if need_more_features:
                    # We need to process more features
                    chunk_start = next_chunk_start + 8
                    chunk_end = min(chunk_start + chunk_size, num_frames)
                    
                    for i in range(chunk_start, chunk_end):
                        mel_idx = i * step_size
                        if mel_idx + context_size > mel_tensor.shape[0]:
                            mel_idx = mel_tensor.shape[0] - context_size
                        
                        mel_window = mel_tensor[mel_idx:mel_idx+context_size].T.unsqueeze(0)  # [1, 80, 16]
                        mel_window = mel_window.unsqueeze(1).to(device)  # [1, 1, 80, 16]
                        
                        with torch.no_grad():
                            audio_feature = model(mel_window)  # [1, 512]
                            audio_features_cache.append(audio_feature.cpu().numpy())
                
                # Skip processing this frame if we don't have enough features yet
                # But don't break out of the loop - continue checking for more features
                if frame_idx >= len(audio_features_cache) - 8:
                    # Prevent infinite loops by checking retry count
                    retry_count += 1
                    if retry_count >= max_retries:
                        logging.warning(f"Reached maximum retries ({max_retries}) waiting for audio features. Terminating.")
                        break
                    # Wait a bit before checking again for streamed features
                    await asyncio.sleep(0.05)
                    continue
                    
                # Reset retry counter if we're processing frames
                retry_count = 0
                
                # Get audio feature with temporal context window
                audio_feat = get_audio_features(np.concatenate(audio_features_cache), frame_idx)
                
                # Reshape to match the expected input
                audio_feat = audio_feat.reshape(1, 32, 16, 16).to(device)

                # Generate the frame with the model
                with torch.no_grad():
                    pred = self.models.net(img_concat_T, audio_feat)[0]
                
                # Convert from tensor to image
                pred = (pred.cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)

                # Apply the generated face to the original image
                final_frame = self.models.img.copy()
                crop_img_out = crop_img_r.copy()
                crop_img_out[4:324, 4:324] = pred
                
                # Resize back to original dimensions
                crop_img_out = cv2.resize(crop_img_out, (w, h), interpolation=cv2.INTER_CUBIC)
                
                # Place the animated face in the original image
                final_frame[self.models.crop_coords[1]:self.models.crop_coords[3], 
                          self.models.crop_coords[0]:self.models.crop_coords[2]] = crop_img_out
                
                # Save debug frame if enabled
                if settings.debug.enabled and settings.debug.save_frames:
                    self._save_debug_frame(final_frame, session_id, frame_idx)
                
                # Encode as JPEG
                _, jpeg = cv2.imencode('.jpg', final_frame)
                
                # Wait if needed to maintain target FPS
                current_time = time.time()
                elapsed = current_time - last_frame_time
                if elapsed < frame_time:
                    await asyncio.sleep(frame_time - elapsed)
                    
                last_frame_time = time.time()
                
                # Yield the frame with HTTP multipart format
                yield (b'--frame\r\n'
                      b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')
                
                # Increment frame counter
                frame_idx += 1
                
                # Log progress at regular intervals
                if frame_idx % 10 == 0:
                    logging.info(f"Generated {frame_idx}/{num_frames} frames (session: {session_id})")
                    
        except Exception as e:
            logging.error(f"Error generating frames from text: {str(e)}", exc_info=True)
            raise

    async def generate_frames_from_audio(self, audio_data: bytes, audio_duration: float = 0.0) -> AsyncGenerator[bytes, None]:
        """
        Generate animated frames from uploaded audio data.
        
        This method processes raw audio data from an uploaded file to extract features
        and generate animated frames showing the template face speaking in sync with
        the provided audio.
        
        Args:
            audio_data: Raw audio bytes from the uploaded file (expected to be 16-bit PCM WAV at 16kHz)
            audio_duration: Optional known audio duration in seconds for better sync
            
        Yields:
            bytes: A series of JPEG-encoded frames with HTTP multipart framing
            
        Raises:
            RuntimeError: If audio processing fails
            ValueError: If audio format is invalid or processing fails
        """
        # Create a unique session ID for this generation
        session_id = f"audio_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
        logging.info(f"Starting frame generation for audio input (session: {session_id})")
        
        # First send a static frame
        yield await self.generate_static_frame()
        
        try:
            # Save the audio for debugging if enabled
            if settings.debug.enabled and settings.debug.save_audio:
                self._save_debug_audio(audio_data, session_id, is_tts=False)
                
            # Convert to numpy array - assume 16-bit PCM audio
            waveform_np = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32767.0
            
            # Get preprocessed image data
            img_concat_T, (h, w), crop_img_r = self._preprocess_image()

            # Convert to mel spectrogram
            mel = melspectrogram(waveform_np).T
            mel_tensor = torch.FloatTensor(mel)
            
            # Process in smaller chunks to enable real-time streaming
            model = self.models.audio_encoder
            device = self.models.device
            
            fps_target = 25  # Target frame rate
            frame_time = 1.0 / fps_target
            last_frame_time = time.time()
            
            # We process in smaller chunks (e.g., 50 frames at a time)
            # This allows frames to start streaming while features are still being extracted
            chunk_size = 50
            step_size = 4  # Step by 4 frames for ~25 fps
            context_size = 16  # Each audio window is 16 frames
            
            # Calculate number of frames
            num_frames = max(1, (mel_tensor.shape[0] - context_size) // step_size + 1)
            
            # Process and stream audio features in chunks
            frame_idx = 0
            audio_features_cache = []  # Cache for already processed features
            
            logging.info(f"Processing {num_frames} frames for audio input (session: {session_id})")
            
            # Process the first chunk of audio features
            chunk_end = min(chunk_size, num_frames)
            for i in range(0, chunk_end):
                mel_idx = i * step_size
                if mel_idx + context_size > mel_tensor.shape[0]:
                    mel_idx = mel_tensor.shape[0] - context_size
                
                mel_window = mel_tensor[mel_idx:mel_idx+context_size].T.unsqueeze(0)  # [1, 80, 16]
                mel_window = mel_window.unsqueeze(1).to(device)  # [1, 1, 80, 16]
                
                with torch.no_grad():
                    audio_feature = model(mel_window)  # [1, 512]
                    audio_features_cache.append(audio_feature.cpu().numpy())
            
            # We need at least 8 frames of context for proper get_audio_features
            # Add padding at the beginning for smooth transitions
            if audio_features_cache:
                first_frame = audio_features_cache[0]
                audio_features_cache = [first_frame] * 8 + audio_features_cache
            
            # Now, process frames chunk by chunk
            max_retries = 10  # Maximum number of retries to prevent infinite loops
            retry_count = 0
            
            while frame_idx < num_frames:
                # Process more audio features if needed
                next_chunk_start = len(audio_features_cache) - 8  # Account for padding
                need_more_features = next_chunk_start + 8 < min(frame_idx + 10, num_frames)
                
                if need_more_features:
                    # We need to process more features
                    chunk_start = next_chunk_start + 8
                    chunk_end = min(chunk_start + chunk_size, num_frames)
                    
                    for i in range(chunk_start, chunk_end):
                        mel_idx = i * step_size
                        if mel_idx + context_size > mel_tensor.shape[0]:
                            mel_idx = mel_tensor.shape[0] - context_size
                        
                        mel_window = mel_tensor[mel_idx:mel_idx+context_size].T.unsqueeze(0)  # [1, 80, 16]
                        mel_window = mel_window.unsqueeze(1).to(device)  # [1, 1, 80, 16]
                        
                        with torch.no_grad():
                            audio_feature = model(mel_window)  # [1, 512]
                            audio_features_cache.append(audio_feature.cpu().numpy())
                
                # Skip processing this frame if we don't have enough features yet
                # But don't break out of the loop - continue checking for more features
                if frame_idx >= len(audio_features_cache) - 8:
                    # Prevent infinite loops by checking retry count
                    retry_count += 1
                    if retry_count >= max_retries:
                        logging.warning(f"Reached maximum retries ({max_retries}) waiting for audio features. Terminating.")
                        break
                    # Wait a bit before checking again for streamed features
                    await asyncio.sleep(0.05)
                    continue
                    
                # Reset retry counter if we're processing frames
                retry_count = 0
                
                # Head motion logic from inference script
                if img_idx > len_img - 1:
                    step_stride = -1
                if img_idx < 1:
                    step_stride = 1
                img_idx += step_stride
                
                # Get the source frame and landmarks for this iteration
                img = self._source_frames[img_idx].copy()
                lms = self._source_landmarks[img_idx]
                
                # Crop the face region
                xmin = lms[1][0]
                ymin = lms[52][1]
                xmax = lms[31][0]
                width = xmax - xmin
                ymax = ymin + width
                
                crop_img = img[ymin:ymax, xmin:xmax]
                h, w = crop_img.shape[:2]
                crop_img = cv2.resize(crop_img, (328, 328), interpolation=cv2.INTER_CUBIC)
                
                img_real_ex = crop_img[4:324, 4:324].copy()
                img_real_ex_ori = img_real_ex.copy()
                img_masked = cv2.rectangle(img_real_ex_ori.copy(), (5, 5, 310, 305), (0, 0, 0), -1)
                
                # Convert to tensors
                img_masked_T = torch.from_numpy((img_masked.transpose(2, 0, 1) / 255.0).astype(np.float32))
                img_real_ex_T = torch.from_numpy((img_real_ex.transpose(2, 0, 1) / 255.0).astype(np.float32))
                img_concat_T = torch.cat([img_real_ex_T, img_masked_T], axis=0)[None].to(device)
                
                # Get audio feature with temporal context window
                audio_feat = get_audio_features(np.concatenate(audio_features_cache), frame_idx)
                
                # Reshape to match the expected input
                audio_feat = audio_feat.reshape(1, 32, 16, 16).to(device)

                # Generate the frame with the model
                with torch.no_grad():
                    pred = self.models.net(img_concat_T, audio_feat)[0]
                
                # Convert from tensor to image
                pred = (pred.cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)

                # Apply the generated face to the original image
                crop_img_out = crop_img.copy()
                crop_img_out[4:324, 4:324] = pred
                
                # Resize back to original dimensions
                crop_img_out = cv2.resize(crop_img_out, (w, h), interpolation=cv2.INTER_CUBIC)
                
                # Place the animated face in the original image
                img[ymin:ymax, xmin:xmax] = crop_img_out
                
                # Save debug frame if enabled
                if settings.debug.enabled and settings.debug.save_frames:
                    self._save_debug_frame(img, session_id, frame_idx)
                
                # Encode as JPEG
                _, jpeg = cv2.imencode('.jpg', img)
                
                # Wait if needed to maintain target FPS
                current_time = time.time()
                elapsed = current_time - last_frame_time
                if elapsed < frame_time:
                    await asyncio.sleep(frame_time - elapsed)
                    
                last_frame_time = time.time()
                
                # Yield the frame with HTTP multipart format
                yield (b'--frame\r\n'
                      b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')
                
                # Increment frame counter
                frame_idx += 1
                
                # Log progress at regular intervals
                if frame_idx % 10 == 0:
                    logging.info(f"Generated {frame_idx}/{num_frames} frames (session: {session_id})")
                    
        except Exception as e:
            logging.error(f"Error generating frames from audio: {str(e)}", exc_info=True)
            raise

class EnhancedFrameGenerationService(FrameGenerationService):
    """
    Enhanced version of FrameGenerationService that adds head motion
    by cycling through different frames from the original video.
    
    This implementation mimics the behavior of the inference script for better animation quality.
    """
    
    def __init__(self, model_service, tts_service):
        """
        Initialize the enhanced frame generation service.
        
        Args:
            model_service: Service that provides models and template image
            tts_service: Service for text-to-speech synthesis
        """
        super().__init__(model_service, tts_service)
        
        # Find the dataset directory for the current model
        self.model_name = self.models.model_name
        self.dataset_dir = os.path.join("./dataset", self.model_name)
        
        # Cache for source frames and landmarks
        self._source_frames = None
        self._source_landmarks = None
        
        # Load source frames and landmarks
        self._load_source_data()
    
    def _load_source_data(self) -> None:
        """
        Load all source frames and landmarks for head motion.
        
        This allows the service to cycle through different frames for more realistic animation.
        """
        img_dir = os.path.join(self.dataset_dir, "full_body_img/")
        lms_dir = os.path.join(self.dataset_dir, "landmarks/")
        
        if not os.path.exists(img_dir) or not os.path.exists(lms_dir):
            logging.warning(f"Source directories not found: {img_dir} or {lms_dir}")
            return
            
        try:
            # Get all image files
            img_files = sorted([f for f in os.listdir(img_dir) if f.endswith('.jpg')], 
                              key=lambda x: int(x.split('.')[0]))
                              
            # Cache for frames and landmarks
            self._source_frames = []
            self._source_landmarks = []
            
            # Load all frames and landmarks
            for img_file in img_files:
                img_path = os.path.join(img_dir, img_file)
                lms_path = os.path.join(lms_dir, img_file.replace('.jpg', '.lms'))
                
                if os.path.exists(img_path) and os.path.exists(lms_path):
                    # Load image
                    img = cv2.imread(img_path)
                    
                    # Load landmarks
                    lms_list = []
                    with open(lms_path, "r") as f:
                        lines = f.read().splitlines()
                        for line in lines:
                            arr = line.split(" ")
                            arr = np.array(arr, dtype=np.float32)
                            lms_list.append(arr)
                    lms = np.array(lms_list, dtype=np.int32)
                    
                    # Store frame and landmarks
                    self._source_frames.append(img)
                    self._source_landmarks.append(lms)
            
            logging.info(f"Loaded {len(self._source_frames)} source frames for head motion")
        except Exception as e:
            logging.error(f"Failed to load source data: {str(e)}", exc_info=True)
            self._source_frames = None
            self._source_landmarks = None

    async def generate_frames(self, text: str, audio_duration: float = 0.0) -> AsyncGenerator[bytes, None]:
        """
        Generate animated frames with head motion from text input.
        
        This method enhances the base implementation by cycling through different
        source frames to create realistic head motion, similar to the inference script.
        
        Args:
            text: The text to synthesize and animate
            audio_duration: Optional known audio duration in seconds for better sync
            
        Yields:
            bytes: A series of JPEG-encoded frames with HTTP multipart framing
            
        Raises:
            RuntimeError: If text-to-speech synthesis fails or audio processing fails
        """
        # If source frames are not available, fall back to the base implementation
        if not self._source_frames or not self._source_landmarks:
            logging.warning("Source frames not available, falling back to static animation")
            async for frame in super().generate_frames(text, audio_duration=audio_duration):
                yield frame
            return
            
        # Create a unique session ID for this generation
        session_id = f"text_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
        logging.info(f"Starting enhanced frame generation for text input (session: {session_id})")
        
        # First send a static frame
        yield await self.generate_static_frame()
        
        try:
            # Synthesize speech
            waveform_bytes = self.tts.synthesize(text)
            if not waveform_bytes:  # Handle empty result
                logging.error("TTS synthesis returned empty result")
                return
                
            # Save the audio for debugging if enabled
            if settings.debug.enabled and settings.debug.save_audio:
                self._save_debug_audio(waveform_bytes, session_id, is_tts=True)
                
            # Convert to numpy array
            waveform_np = np.frombuffer(waveform_bytes, dtype=np.int16).astype(np.float32) / 32767.0

            # Convert to mel spectrogram
            mel = melspectrogram(waveform_np).T
            mel_tensor = torch.FloatTensor(mel)
            
            # Process in smaller chunks to enable real-time streaming
            model = self.models.audio_encoder
            device = self.models.device
            
            # Calculate an adjusted frame rate based on audio duration
            fps_target = self._calculate_adjusted_fps(mel_tensor, audio_duration)
            frame_time = 1.0 / fps_target
            last_frame_time = time.time()
            
            # We process in smaller chunks (e.g., 50 frames at a time)
            # This allows frames to start streaming while features are still being extracted
            chunk_size = 50
            step_size = 4  # Step by 4 frames for ~25 fps
            context_size = 16  # Each audio window is 16 frames
            
            # Calculate number of frames
            num_frames = max(1, (mel_tensor.shape[0] - context_size) // step_size + 1)
            
            # Process and stream audio features in chunks
            frame_idx = 0
            audio_features_cache = []  # Cache for already processed features
            
            logging.info(f"Processing {num_frames} frames for text input (session: {session_id})")
            
            # Process the first chunk of audio features
            chunk_end = min(chunk_size, num_frames)
            for i in range(0, chunk_end):
                mel_idx = i * step_size
                if mel_idx + context_size > mel_tensor.shape[0]:
                    mel_idx = mel_tensor.shape[0] - context_size
                
                mel_window = mel_tensor[mel_idx:mel_idx+context_size].T.unsqueeze(0)  # [1, 80, 16]
                mel_window = mel_window.unsqueeze(1).to(device)  # [1, 1, 80, 16]
                
                with torch.no_grad():
                    audio_feature = model(mel_window)  # [1, 512]
                    audio_features_cache.append(audio_feature.cpu().numpy())
            
            # We need at least 8 frames of context for proper get_audio_features
            # Add padding at the beginning for smooth transitions
            if audio_features_cache:
                first_frame = audio_features_cache[0]
                audio_features_cache = [first_frame] * 8 + audio_features_cache
            
            # Variables for head motion
            step_stride = 0
            img_idx = 0
            len_img = len(self._source_frames) - 1
            
            # Now, process frames chunk by chunk
            max_retries = 10  # Maximum number of retries to prevent infinite loops
            retry_count = 0
            
            while frame_idx < num_frames:
                # Process more audio features if needed
                next_chunk_start = len(audio_features_cache) - 8  # Account for padding
                need_more_features = next_chunk_start + 8 < min(frame_idx + 10, num_frames)
                
                if need_more_features:
                    # We need to process more features
                    chunk_start = next_chunk_start + 8
                    chunk_end = min(chunk_start + chunk_size, num_frames)
                    
                    for i in range(chunk_start, chunk_end):
                        mel_idx = i * step_size
                        if mel_idx + context_size > mel_tensor.shape[0]:
                            mel_idx = mel_tensor.shape[0] - context_size
                        
                        mel_window = mel_tensor[mel_idx:mel_idx+context_size].T.unsqueeze(0)  # [1, 80, 16]
                        mel_window = mel_window.unsqueeze(1).to(device)  # [1, 1, 80, 16]
                        
                        with torch.no_grad():
                            audio_feature = model(mel_window)  # [1, 512]
                            audio_features_cache.append(audio_feature.cpu().numpy())
                
                # Skip processing this frame if we don't have enough features yet
                # But don't break out of the loop - continue checking for more features
                if frame_idx >= len(audio_features_cache) - 8:
                    # Prevent infinite loops by checking retry count
                    retry_count += 1
                    if retry_count >= max_retries:
                        logging.warning(f"Reached maximum retries ({max_retries}) waiting for audio features. Terminating.")
                        break
                    # Wait a bit before checking again for streamed features
                    await asyncio.sleep(0.05)
                    continue
                    
                # Reset retry counter if we're processing frames
                retry_count = 0
                
                # Head motion logic from inference script
                if img_idx > len_img - 1:
                    step_stride = -1
                if img_idx < 1:
                    step_stride = 1
                img_idx += step_stride
                
                # Get the source frame and landmarks for this iteration
                img = self._source_frames[img_idx].copy()
                lms = self._source_landmarks[img_idx]
                
                # Crop the face region
                xmin = lms[1][0]
                ymin = lms[52][1]
                xmax = lms[31][0]
                width = xmax - xmin
                ymax = ymin + width
                
                crop_img = img[ymin:ymax, xmin:xmax]
                h, w = crop_img.shape[:2]
                crop_img = cv2.resize(crop_img, (328, 328), interpolation=cv2.INTER_CUBIC)
                
                img_real_ex = crop_img[4:324, 4:324].copy()
                img_real_ex_ori = img_real_ex.copy()
                img_masked = cv2.rectangle(img_real_ex_ori.copy(), (5, 5, 310, 305), (0, 0, 0), -1)
                
                # Convert to tensors
                img_masked_T = torch.from_numpy((img_masked.transpose(2, 0, 1) / 255.0).astype(np.float32))
                img_real_ex_T = torch.from_numpy((img_real_ex.transpose(2, 0, 1) / 255.0).astype(np.float32))
                img_concat_T = torch.cat([img_real_ex_T, img_masked_T], axis=0)[None].to(device)
                
                # Get audio feature with temporal context window
                audio_feat = get_audio_features(np.concatenate(audio_features_cache), frame_idx)
                
                # Reshape to match the expected input
                audio_feat = audio_feat.reshape(1, 32, 16, 16).to(device)

                # Generate the frame with the model
                with torch.no_grad():
                    pred = self.models.net(img_concat_T, audio_feat)[0]
                
                # Convert from tensor to image
                pred = (pred.cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)

                # Apply the generated face to the original image
                crop_img_out = crop_img.copy()
                crop_img_out[4:324, 4:324] = pred
                
                # Resize back to original dimensions
                crop_img_out = cv2.resize(crop_img_out, (w, h), interpolation=cv2.INTER_CUBIC)
                
                # Place the animated face in the original image
                img[ymin:ymax, xmin:xmax] = crop_img_out
                
                # Save debug frame if enabled
                if settings.debug.enabled and settings.debug.save_frames:
                    self._save_debug_frame(img, session_id, frame_idx)
                
                # Encode as JPEG
                _, jpeg = cv2.imencode('.jpg', img)
                
                # Wait if needed to maintain target FPS
                current_time = time.time()
                elapsed = current_time - last_frame_time
                if elapsed < frame_time:
                    await asyncio.sleep(frame_time - elapsed)
                    
                last_frame_time = time.time()
                
                # Yield the frame with HTTP multipart format
                yield (b'--frame\r\n'
                      b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')
                
                # Increment frame counter
                frame_idx += 1
                
                # Log progress at regular intervals
                if frame_idx % 10 == 0:
                    logging.info(f"Generated {frame_idx}/{num_frames} frames (session: {session_id})")
                    
        except Exception as e:
            logging.error(f"Error generating frames from text: {str(e)}", exc_info=True)
            raise
            
    async def generate_frames_from_audio(self, audio_data: bytes, audio_duration: float = 0.0) -> AsyncGenerator[bytes, None]:
        """
        Generate animated frames with head motion from uploaded audio data.
        
        This method enhances the base implementation by cycling through different
        source frames to create realistic head motion, similar to the inference script.
        
        Args:
            audio_data: Raw audio bytes from the uploaded file
            audio_duration: Optional known audio duration in seconds for better sync
            
        Yields:
            bytes: A series of JPEG-encoded frames with HTTP multipart framing
            
        Raises:
            RuntimeError: If audio processing fails
        """
        # If source frames are not available, fall back to the base implementation
        if not self._source_frames or not self._source_landmarks:
            logging.warning("Source frames not available, falling back to static animation")
            async for frame in super().generate_frames_from_audio(audio_data, audio_duration=audio_duration):
                yield frame
            return
            
        # Create a unique session ID for this generation
        session_id = f"audio_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
        logging.info(f"Starting enhanced frame generation for audio input (session: {session_id})")
        
        # First send a static frame
        yield await self.generate_static_frame()
        
        try:
            # Save the audio for debugging if enabled
            if settings.debug.enabled and settings.debug.save_audio:
                self._save_debug_audio(audio_data, session_id, is_tts=False)
                
            # Convert to numpy array - assume 16-bit PCM audio
            waveform_np = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32767.0

            # Convert to mel spectrogram
            mel = melspectrogram(waveform_np).T
            mel_tensor = torch.FloatTensor(mel)
            
            # Process in smaller chunks to enable real-time streaming
            model = self.models.audio_encoder
            device = self.models.device
            
            fps_target = 25  # Target frame rate
            frame_time = 1.0 / fps_target
            last_frame_time = time.time()
            
            # We process in smaller chunks (e.g., 50 frames at a time)
            # This allows frames to start streaming while features are still being extracted
            chunk_size = 50
            step_size = 4  # Step by 4 frames for ~25 fps
            context_size = 16  # Each audio window is 16 frames
            
            # Calculate number of frames
            num_frames = max(1, (mel_tensor.shape[0] - context_size) // step_size + 1)
            
            # Process and stream audio features in chunks
            frame_idx = 0
            audio_features_cache = []  # Cache for already processed features
            
            logging.info(f"Processing {num_frames} frames for audio input (session: {session_id})")
            
            # Process the first chunk of audio features
            chunk_end = min(chunk_size, num_frames)
            for i in range(0, chunk_end):
                mel_idx = i * step_size
                if mel_idx + context_size > mel_tensor.shape[0]:
                    mel_idx = mel_tensor.shape[0] - context_size
                
                mel_window = mel_tensor[mel_idx:mel_idx+context_size].T.unsqueeze(0)  # [1, 80, 16]
                mel_window = mel_window.unsqueeze(1).to(device)  # [1, 1, 80, 16]
                
                with torch.no_grad():
                    audio_feature = model(mel_window)  # [1, 512]
                    audio_features_cache.append(audio_feature.cpu().numpy())
            
            # We need at least 8 frames of context for proper get_audio_features
            # Add padding at the beginning for smooth transitions
            if audio_features_cache:
                first_frame = audio_features_cache[0]
                audio_features_cache = [first_frame] * 8 + audio_features_cache
            
            # Now, process frames chunk by chunk
            max_retries = 10  # Maximum number of retries to prevent infinite loops
            retry_count = 0
            
            while frame_idx < num_frames:
                # Process more audio features if needed
                next_chunk_start = len(audio_features_cache) - 8  # Account for padding
                need_more_features = next_chunk_start + 8 < min(frame_idx + 10, num_frames)
                
                if need_more_features:
                    # We need to process more features
                    chunk_start = next_chunk_start + 8
                    chunk_end = min(chunk_start + chunk_size, num_frames)
                    
                    for i in range(chunk_start, chunk_end):
                        mel_idx = i * step_size
                        if mel_idx + context_size > mel_tensor.shape[0]:
                            mel_idx = mel_tensor.shape[0] - context_size
                        
                        mel_window = mel_tensor[mel_idx:mel_idx+context_size].T.unsqueeze(0)  # [1, 80, 16]
                        mel_window = mel_window.unsqueeze(1).to(device)  # [1, 1, 80, 16]
                        
                        with torch.no_grad():
                            audio_feature = model(mel_window)  # [1, 512]
                            audio_features_cache.append(audio_feature.cpu().numpy())
                
                # Skip processing this frame if we don't have enough features yet
                # But don't break out of the loop - continue checking for more features
                if frame_idx >= len(audio_features_cache) - 8:
                    # Prevent infinite loops by checking retry count
                    retry_count += 1
                    if retry_count >= max_retries:
                        logging.warning(f"Reached maximum retries ({max_retries}) waiting for audio features. Terminating.")
                        break
                    # Wait a bit before checking again for streamed features
                    await asyncio.sleep(0.05)
                    continue
                    
                # Reset retry counter if we're processing frames
                retry_count = 0
                
                # Head motion logic from inference script
                if img_idx > len_img - 1:
                    step_stride = -1
                if img_idx < 1:
                    step_stride = 1
                img_idx += step_stride
                
                # Get the source frame and landmarks for this iteration
                img = self._source_frames[img_idx].copy()
                lms = self._source_landmarks[img_idx]
                
                # Crop the face region
                xmin = lms[1][0]
                ymin = lms[52][1]
                xmax = lms[31][0]
                width = xmax - xmin
                ymax = ymin + width
                
                crop_img = img[ymin:ymax, xmin:xmax]
                h, w = crop_img.shape[:2]
                crop_img = cv2.resize(crop_img, (328, 328), interpolation=cv2.INTER_CUBIC)
                
                img_real_ex = crop_img[4:324, 4:324].copy()
                img_real_ex_ori = img_real_ex.copy()
                img_masked = cv2.rectangle(img_real_ex_ori.copy(), (5, 5, 310, 305), (0, 0, 0), -1)
                
                # Convert to tensors
                img_masked_T = torch.from_numpy((img_masked.transpose(2, 0, 1) / 255.0).astype(np.float32))
                img_real_ex_T = torch.from_numpy((img_real_ex.transpose(2, 0, 1) / 255.0).astype(np.float32))
                img_concat_T = torch.cat([img_real_ex_T, img_masked_T], axis=0)[None].to(device)
                
                # Get audio feature with temporal context window
                audio_feat = get_audio_features(np.concatenate(audio_features_cache), frame_idx)
                
                # Reshape to match the expected input
                audio_feat = audio_feat.reshape(1, 32, 16, 16).to(device)

                # Generate the frame with the model
                with torch.no_grad():
                    pred = self.models.net(img_concat_T, audio_feat)[0]
                
                # Convert from tensor to image
                pred = (pred.cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)

                # Apply the generated face to the original image
                crop_img_out = crop_img.copy()
                crop_img_out[4:324, 4:324] = pred
                
                # Resize back to original dimensions
                crop_img_out = cv2.resize(crop_img_out, (w, h), interpolation=cv2.INTER_CUBIC)
                
                # Place the animated face in the original image
                img[ymin:ymax, xmin:xmax] = crop_img_out
                
                # Save debug frame if enabled
                if settings.debug.enabled and settings.debug.save_frames:
                    self._save_debug_frame(img, session_id, frame_idx)
                
                # Encode as JPEG
                _, jpeg = cv2.imencode('.jpg', img)
                
                # Wait if needed to maintain target FPS
                current_time = time.time()
                elapsed = current_time - last_frame_time
                if elapsed < frame_time:
                    await asyncio.sleep(frame_time - elapsed)
                    
                last_frame_time = time.time()
                
                # Yield the frame with HTTP multipart format
                yield (b'--frame\r\n'
                      b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')
                
                # Increment frame counter
                frame_idx += 1
                
                # Log progress at regular intervals
                if frame_idx % 10 == 0:
                    logging.info(f"Generated {frame_idx}/{num_frames} frames (session: {session_id})")
                    
        except Exception as e:
            logging.error(f"Error generating frames from audio: {str(e)}", exc_info=True)
            raise

    async def generate_dynamic_avatar(self) -> AsyncGenerator[bytes, None]:
        """
        Generate a dynamic avatar with natural head movement but no speech.
        
        This provides a more natural looking idle state for the avatar when
        not speaking, cycling through source frames to create subtle head movements.
        
        Yields:
            bytes: A series of JPEG-encoded frames with HTTP multipart framing
        """
        # If source frames are not available, fall back to the static frame
        if not self._source_frames or not self._source_landmarks:
            logging.warning("Source frames not available, falling back to static avatar")
            yield await self.generate_static_frame()
            return
        
        # Create a unique session ID for this generation
        session_id = f"avatar_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
        logging.info(f"Starting dynamic avatar generation (session: {session_id})")
        
        try:
            fps_target = 15  # Lower frame rate for idle animation
            frame_time = 1.0 / fps_target
            last_frame_time = time.time()
            
            # Variables for head motion
            step_stride = 0
            img_idx = 0
            len_img = len(self._source_frames) - 1
            
            # For an indefinite idle animation, we need to yield frames continuously
            frame_idx = 0
            
            while True:
                # Head motion logic from inference script
                if img_idx > len_img - 1:
                    step_stride = -1
                if img_idx < 1:
                    step_stride = 1
                img_idx += step_stride
                
                # Get the source frame and landmarks for this iteration
                img = self._source_frames[img_idx].copy()
                
                # Save debug frame if enabled
                if settings.debug.enabled and settings.debug.save_frames and frame_idx % 10 == 0:
                    self._save_debug_frame(img, session_id, frame_idx)
                
                # Encode as high-quality JPEG
                jpeg = self._encode_high_quality_jpeg(img)
                
                # Wait if needed to maintain target FPS
                current_time = time.time()
                elapsed = current_time - last_frame_time
                if elapsed < frame_time:
                    await asyncio.sleep(frame_time - elapsed)
                    
                last_frame_time = time.time()
                
                # Yield the frame with HTTP multipart format
                yield (b'--frame\r\n'
                      b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')
                
                # Increment frame counter (for debugging purposes)
                frame_idx += 1
                
        except Exception as e:
            logging.error(f"Error generating dynamic avatar: {str(e)}", exc_info=True)
            # If an error occurs, try to fall back to static frame
            yield await self.generate_static_frame()

    # Helper method for adjusting frame rate based on audio duration
    def _calculate_adjusted_fps(self, mel_tensor, audio_duration: float) -> float:
        """
        Calculate an adjusted frame rate based on the audio duration and mel spectrogram size.
        
        This ensures the video animation synchronizes with the audio duration.
        
        Args:
            mel_tensor: The mel spectrogram tensor
            audio_duration: The audio duration in seconds
            
        Returns:
            float: The adjusted frames per second value
        """
        if audio_duration <= 0:
            return 25.0  # Default frame rate
            
        # Calculate how many frames we'll generate
        step_size = 4  # Step by 4 frames for ~25 fps
        context_size = 16  # Each audio window is 16 frames
        estimated_frames = max(1, (mel_tensor.shape[0] - context_size) // step_size + 1)
        
        # Adjust fps to match audio duration (with a small buffer)
        adjusted_fps = estimated_frames / (audio_duration * 0.98)  # 98% of duration to ensure we don't end too early
        
        # Limit to reasonable values (15-30 fps)
        adjusted_fps = max(15, min(30, adjusted_fps))
        
        logging.info(f"Adjusting frame rate to {adjusted_fps:.2f} fps to match audio duration of {audio_duration:.2f} seconds")
        return adjusted_fps

    def _encode_high_quality_jpeg(self, img):
        """
        Encode an image as a high-quality JPEG.
        
        Args:
            img: The image to encode
            
        Returns:
            bytes: The JPEG-encoded image
        """
        # Use high-quality JPEG encoding
        encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), 95]
        _, jpeg = cv2.imencode('.jpg', img, encode_params)
        return jpeg
