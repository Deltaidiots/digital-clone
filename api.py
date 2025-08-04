import uvicorn
import logging
import sys
import os
import fastapi
from fastapi import FastAPI, Depends, HTTPException, UploadFile, File, BackgroundTasks
from fastapi.responses import StreamingResponse, FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Configure logging
logging.basicConfig(
    level=logging.INFO,  # Standard logging level for production
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Import our services
from services.models import model_service, ModelService
from services.tts import tts_service, TTSService
from services.generator import FrameGenerationService, EnhancedFrameGenerationService
from config import settings

app = FastAPI(title="SyncTalk_2D API")

# Add CORS middleware to allow frontend requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, you'd restrict this to your frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add error handler for service initialization failures
@app.exception_handler(500)
async def service_error_handler(request, exc):
    # Handle both FastAPI HTTPException (with detail attribute) and standard Python exceptions
    error_message = str(exc.detail) if hasattr(exc, 'detail') else str(exc)
    return JSONResponse(
        status_code=500,
        content={
            "error": error_message,
            "message": "Service initialization failed. Check server logs for details."
        }
    )

# Add health check endpoint
@app.get("/health")
async def health_check():
    # Check if TTS is working by attempting to synthesize a very short text
    tts_status = "ok"
    try:
        test_audio = tts_service.synthesize("test")
        if not test_audio or len(test_audio) < 100:
            tts_status = "fallback"
    except:
        tts_status = "error"
    
    health = {
        "status": "ok" if tts_status == "ok" else "degraded",
        "services": {
            "tts": tts_status,
            "model": "ok"  # Assume model service is ok for simplicity
        }
    }
    return health

# --- Dependency Injection ---
# This function will be called for each request to get the generator service
def get_generator_service():
    # Use the enhanced generator service for better animation quality
    try:
        # First try to use the enhanced service with head motion
        return EnhancedFrameGenerationService(model_service, tts_service)
    except Exception as e:
        logging.warning(f"Failed to initialize EnhancedFrameGenerationService: {str(e)}. Falling back to basic service.")
        # Fall back to the original service if enhanced service fails
        return FrameGenerationService(model_service, tts_service)

# Request model for text-to-speech generation
class TextRequest(BaseModel):
    text: str

@app.get("/generate")
async def generate(
    text: str,
    duration: float = 0.0,  # Audio duration in seconds, used for sync
    generator: FrameGenerationService = Depends(get_generator_service)
):
    """
    Generate animated face video frames from text input.
    
    Args:
        text: The text to synthesize and animate (query parameter)
        duration: Optional audio duration in seconds for better sync
        generator: Frame generation service (injected)
        
    Returns:
        StreamingResponse: A stream of JPEG frames
    """
    # Make sure we're properly handling the async generator
    async def stream_generator():
        async for frame in generator.generate_frames(text, audio_duration=duration):
            yield frame
            
    return StreamingResponse(
        stream_generator(),
        media_type='multipart/x-mixed-replace; boundary=frame'
    )

@app.post("/generate")
async def generate_post(
    request: TextRequest,
    duration: float = 0.0,  # Audio duration in seconds, used for sync
    generator: FrameGenerationService = Depends(get_generator_service)
):
    """
    Generate animated face video frames from text input (POST method).
    
    Args:
        request: Contains the text to synthesize and animate
        duration: Optional audio duration in seconds for better sync
        generator: Frame generation service (injected)
        
    Returns:
        StreamingResponse: A stream of JPEG frames
    """
    # Make sure we're properly handling the async generator
    async def stream_generator():
        async for frame in generator.generate_frames(request.text, audio_duration=duration):
            yield frame
            
    return StreamingResponse(
        stream_generator(),
        media_type='multipart/x-mixed-replace; boundary=frame'
    )

@app.get("/avatar")
async def avatar(
    generator: FrameGenerationService = Depends(get_generator_service)
):
    """
    Return a static avatar frame.
    
    Args:
        generator: Frame generation service (injected)
        
    Returns:
        StreamingResponse: A static JPEG frame of the avatar
    """
    async def frame_generator():
        yield await generator.generate_static_frame()
        
    return StreamingResponse(
        frame_generator(),
        media_type='multipart/x-mixed-replace; boundary=frame'
    )

# New endpoint for generating video from audio file upload
@app.post("/generate-from-audio")
async def generate_from_audio(
    audio_file: UploadFile = File(...),
    generator: FrameGenerationService = Depends(get_generator_service)
) -> StreamingResponse:
    """
    Generate animated face video frames from an uploaded audio file.
    
    Args:
        audio_file: The audio file to process (must be 16-bit PCM WAV at 16kHz)
        generator: Frame generation service (injected)
        
    Returns:
        StreamingResponse: A stream of JPEG frames
    
    Raises:
        HTTPException: If there is an error processing the audio file
    """
    try:
        # Check file format - only accept WAV files for now
        if not audio_file.filename.lower().endswith('.wav'):
            raise ValueError("Only WAV audio files are currently supported")
            
        # Read the audio file
        audio_data = await audio_file.read()
        
        if len(audio_data) < 1024:  # Basic check for empty/corrupted files
            raise ValueError("Audio file appears to be empty or corrupted")
            
        # Stream the animated frames
        # We need to make sure we're returning an async generator, not a coroutine
        async def stream_generator():
            async for frame in generator.generate_frames_from_audio(audio_data):
                yield frame
                
        return StreamingResponse(
            stream_generator(),
            media_type='multipart/x-mixed-replace; boundary=frame'
        )
    except Exception as e:
        logging.error(f"Error processing audio file: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error processing audio: {str(e)}")

# New GET endpoint for generate-from-audio to support direct URL streaming
@app.get("/generate-from-audio")
async def generate_from_audio_get(
    generator: FrameGenerationService = Depends(get_generator_service)
) -> StreamingResponse:
    """
    Stream a static avatar frame when directly accessing the endpoint via GET.
    This allows the <img> tag to use this as a src and then the form POST
    will update the stream with animation frames.
    
    Args:
        generator: Frame generation service (injected)
        
    Returns:
        StreamingResponse: A stream with the static avatar frame
    """
    # Return the static avatar frame to initialize the stream
    async def frame_generator():
        yield await generator.generate_static_frame()
        
    return StreamingResponse(
        frame_generator(),
        media_type='multipart/x-mixed-replace; boundary=frame'
    )

# Serve static files
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
async def root(): 
    """ Serve the main HTML page. """
    return FileResponse('static/index.html')

# Debug control endpoints
class DebugConfig(BaseModel):
    """Request model for debug configuration"""
    enabled: bool
    save_frames: bool
    save_audio: bool

@app.get("/debug/status")
async def get_debug_status():
    """
    Get the current debug configuration.
    
    Returns:
        dict: The current debug configuration settings
    """
    return {
        "enabled": settings.debug.enabled,
        "save_frames": settings.debug.save_frames,
        "save_audio": settings.debug.save_audio,
        "frames_dir": settings.debug.frames_dir,
        "audio_dir": settings.debug.audio_dir
    }

@app.post("/debug/config")
async def set_debug_config(config: DebugConfig):
    """
    Update the debug configuration.
    
    Args:
        config: The new debug configuration settings
        
    Returns:
        dict: The updated debug configuration
    """
    settings.debug.enabled = config.enabled
    settings.debug.save_frames = config.save_frames
    settings.debug.save_audio = config.save_audio
    
    # Create directories if needed
    if config.enabled:
        if config.save_frames and not os.path.exists(settings.debug.frames_dir):
            os.makedirs(settings.debug.frames_dir, exist_ok=True)
            
        if config.save_audio and not os.path.exists(settings.debug.audio_dir):
            os.makedirs(settings.debug.audio_dir, exist_ok=True)
    
    return {
        "enabled": settings.debug.enabled,
        "save_frames": settings.debug.save_frames,
        "save_audio": settings.debug.save_audio,
        "frames_dir": settings.debug.frames_dir,
        "audio_dir": settings.debug.audio_dir
    }

@app.get("/get-tts-audio")
async def get_tts_audio(
    text: str
) -> FileResponse:
    """
    Generate TTS audio from text and return it as a downloadable WAV file.
    
    Args:
        text: The text to synthesize
        
    Returns:
        FileResponse: WAV audio file response
    """
    try:
        # Get the raw audio data from the TTS service
        raw_audio = tts_service.synthesize(text)
        if not raw_audio:
            raise HTTPException(status_code=500, detail="Failed to synthesize speech")
        
        # Calculate audio duration
        audio_duration = tts_service.get_audio_duration(raw_audio)
        
        # Create a WAV file in memory
        import io
        import wave
        
        wav_bytes = io.BytesIO()
        with wave.open(wav_bytes, "wb") as wav_file:
            wav_file.setnchannels(1)  # Mono
            wav_file.setsampwidth(2)  # 16-bit
            wav_file.setframerate(tts_service.sample_rate)  # Sample rate (typically 16000)
            wav_file.writeframes(raw_audio)
        
        # Reset the position to the beginning of the BytesIO object
        wav_bytes.seek(0)
        
        # Create a temporary file for the FileResponse
        import tempfile
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        temp_file.write(wav_bytes.read())
        temp_file_path = temp_file.name
        temp_file.close()
        
        # Return the audio file with appropriate headers, including the duration
        return FileResponse(
            path=temp_file_path,
            media_type="audio/wav",
            headers={
                "Content-Disposition": "attachment; filename=tts_audio.wav",
                "X-Audio-Duration": str(audio_duration)  # Add duration as header
            },
            filename="tts_audio.wav",
            background=fastapi.BackgroundTasks()
        )
    except Exception as e:
        logging.error(f"Error generating TTS audio: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error generating TTS audio: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
