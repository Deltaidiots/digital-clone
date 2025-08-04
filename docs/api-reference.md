# API Reference

## Overview

![API Architecture](diagrams/api-architecture.puml)

SyncTalk_2D provides a comprehensive REST API built with FastAPI for generating talking head animations. This document covers all available endpoints, request/response formats, and usage examples.

## Base URL

```
http://localhost:8000
```

## Authentication

Currently, no authentication is required. The API is designed for local development and deployment.

## Endpoints

### 1. Web Interface

#### `GET /`

Returns the main web interface for interactive usage.

**Response**: HTML page with the web interface

**Example**:
```bash
curl http://localhost:8000/
```

---

### 2. Text-to-Speech Generation

#### `GET /generate`

Generate animation from text using query parameters (streaming response).

**Parameters**:
- `text` (string, required): Text to convert to speech and animate
- `character` (string, optional): Character name (default: from config)

**Response**: Server-Sent Events (SSE) stream of base64-encoded frames

**Example**:
```bash
curl "http://localhost:8000/generate?text=Hello%20world&character=Awais"
```

**Response Format**:
```
data: {"frame": "base64_encoded_frame_data", "frame_number": 1}

data: {"frame": "base64_encoded_frame_data", "frame_number": 2}

...
```

#### `POST /generate`

Generate animation from text using JSON body.

**Request Body**:
```json
{
  "text": "Hello, this is a test message",
  "character": "Awais"
}
```

**Response**: Same SSE stream as GET version

**Example**:
```bash
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello world", "character": "Awais"}'
```

---

### 3. Audio File Generation

#### `POST /generate-from-audio`

Generate animation from uploaded audio file.

**Request**: Multipart form data
- `audio` (file, required): WAV audio file
- `character` (string, optional): Character name

**Response**: SSE stream of base64-encoded frames

**Example**:
```bash
curl -X POST http://localhost:8000/generate-from-audio \
  -F "audio=@path/to/audio.wav" \
  -F "character=Awais"
```

---

### 4. Static Assets

#### `GET /avatar`

Get a static avatar frame for the specified character.

**Parameters**:
- `character` (string, optional): Character name (default: from config)

**Response**: JPEG image

**Example**:
```bash
curl "http://localhost:8000/avatar?character=Awais" -o avatar.jpg
```

---

### 5. Audio Services

#### `GET /get-tts-audio`

Generate and download TTS audio without video animation.

**Parameters**:
- `text` (string, required): Text to convert to speech

**Response**: WAV audio file

**Example**:
```bash
curl "http://localhost:8000/get-tts-audio?text=Hello%20world" -o audio.wav
```

---

### 6. Health Check

#### `GET /health`

Check the health status of the API and its dependencies.

**Response**:
```json
{
  "status": "healthy",
  "services": {
    "model": "loaded",
    "tts": "available",
    "gpu": "cuda:0"
  },
  "timestamp": "2025-08-04T12:00:00Z"
}
```

**Example**:
```bash
curl http://localhost:8000/health
```

---

### 7. Debug Endpoints

#### `GET /debug/status`

Get current debug configuration and status.

**Response**:
```json
{
  "debug_enabled": false,
  "save_frames": false,
  "save_audio": false,
  "frames_dir": "./debug/frames",
  "audio_dir": "./debug/audio"
}
```

#### `POST /debug/config`

Update debug configuration.

**Request Body**:
```json
{
  "enabled": true,
  "save_frames": true,
  "save_audio": true
}
```

**Response**:
```json
{
  "status": "updated",
  "config": {
    "enabled": true,
    "save_frames": true,
    "save_audio": true
  }
}
```

**Example**:
```bash
curl -X POST http://localhost:8000/debug/config \
  -H "Content-Type: application/json" \
  -d '{"enabled": true, "save_frames": true}'
```

## Response Formats

### Streaming Responses

The main generation endpoints use Server-Sent Events (SSE) for real-time streaming:

```
Content-Type: text/event-stream
Cache-Control: no-cache
Connection: keep-alive

data: {"frame": "base64_frame_data", "frame_number": 1}

data: {"frame": "base64_frame_data", "frame_number": 2}

data: {"status": "complete", "total_frames": 150}
```

### Error Responses

All endpoints return consistent error formats:

```json
{
  "error": "error_type",
  "message": "Human readable error message",
  "details": {
    "additional": "context_information"
  }
}
```

Common HTTP status codes:
- `400`: Bad Request (invalid parameters)
- `404`: Not Found (character not found)
- `500`: Internal Server Error (model/processing error)

## Client Examples

### Python Client

```python
import requests
import json
import base64
from PIL import Image
import io

def generate_animation(text, character="Awais"):
    """Generate animation and save frames"""
    url = "http://localhost:8000/generate"
    params = {"text": text, "character": character}
    
    response = requests.get(url, params=params, stream=True)
    
    frames = []
    for line in response.iter_lines():
        if line.startswith(b'data: '):
            data = json.loads(line[6:])
            if 'frame' in data:
                # Decode base64 frame
                frame_data = base64.b64decode(data['frame'])
                frame = Image.open(io.BytesIO(frame_data))
                frames.append(frame)
    
    return frames

# Usage
frames = generate_animation("Hello, how are you today?")
print(f"Generated {len(frames)} frames")
```

### JavaScript Client

```javascript
async function generateAnimation(text, character = "Awais") {
    const url = `http://localhost:8000/generate?text=${encodeURIComponent(text)}&character=${character}`;
    
    const response = await fetch(url);
    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    
    const frames = [];
    
    while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        
        const chunk = decoder.decode(value);
        const lines = chunk.split('\n');
        
        for (const line of lines) {
            if (line.startsWith('data: ')) {
                const data = JSON.parse(line.substring(6));
                if (data.frame) {
                    frames.push(data.frame);
                }
            }
        }
    }
    
    return frames;
}

// Usage
generateAnimation("Hello world").then(frames => {
    console.log(`Generated ${frames.length} frames`);
});
```

### cURL Examples

#### Basic text generation:
```bash
curl "http://localhost:8000/generate?text=Hello%20world"
```

#### Upload audio file:
```bash
curl -X POST http://localhost:8000/generate-from-audio \
  -F "audio=@speech.wav" \
  -F "character=May"
```

#### Enable debugging:
```bash
curl -X POST http://localhost:8000/debug/config \
  -H "Content-Type: application/json" \
  -d '{"enabled": true, "save_frames": true, "save_audio": true}'
```

## Rate Limiting

Currently, no rate limiting is implemented. For production deployment, consider adding:
- Request rate limits per IP
- Concurrent generation limits
- Resource usage monitoring

## Configuration

The API behavior can be configured through `config.py`:

```python
# config.py
class AppConfig:
    models: ModelConfig = Field(default_factory=ModelConfig)
    tts: TTSConfig = Field(default_factory=TTSConfig)
    debug: DebugConfig = Field(default_factory=DebugConfig)

# Access in API
from config import settings
device = settings.models.device
```

## Deployment Considerations

### Production Deployment

```bash
# Install production dependencies
pip install gunicorn

# Run with Gunicorn
gunicorn api:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
```

### Docker Deployment

```dockerfile
FROM python:3.10-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 8000

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Performance Optimization

- Use GPU acceleration when available
- Implement connection pooling for multiple clients
- Add caching for frequently requested static assets
- Monitor memory usage and implement cleanup
- Use async processing for better concurrency
