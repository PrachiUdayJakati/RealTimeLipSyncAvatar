"""
Real-Time Lipsync Avatar Web Application
FastAPI-based web server optimized for 16GB GPU VRAM
"""

import os
import logging
import asyncio
import tempfile
import time
from pathlib import Path
from typing import Optional, Dict, Any, List
import uuid

# FastAPI and web dependencies
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, BackgroundTasks
from fastapi.responses import HTMLResponse, FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from starlette.requests import Request
import uvicorn

# AI and processing
import torch
from models.model_manager import ModelManager
from utils.tts_pipeline import TTSPipeline

# Configuration
from dotenv import load_dotenv
load_dotenv()

# Configure logging
logging.basicConfig(
    level=getattr(logging, os.getenv("LOG_LEVEL", "INFO")),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Real-Time Lipsync Avatar",
    description="AI-powered real-time lipsync video generation",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static files and templates
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Global instances
model_manager: Optional[ModelManager] = None
tts_pipeline: Optional[TTSPipeline] = None

# Configuration
MAX_UPLOAD_SIZE = int(os.getenv("MAX_UPLOAD_SIZE_MB", "100")) * 1024 * 1024  # 100MB
UPLOAD_DIR = Path("uploads")
OUTPUT_DIR = Path("outputs")

# Ensure directories exist
for directory in [UPLOAD_DIR / "images", UPLOAD_DIR / "audio", OUTPUT_DIR / "videos", "static", "templates"]:
    directory.mkdir(parents=True, exist_ok=True)

@app.on_event("startup")
async def startup_event():
    """Initialize AI models and services on startup"""
    global model_manager, tts_pipeline

    logger.info("🚀 Starting Real-Time Lipsync Avatar application...")

    # Initialize TTS pipeline
    logger.info("Initializing TTS pipeline...")
    tts_pipeline = TTSPipeline()

    # Initialize model manager
    logger.info("Initializing AI model manager...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    max_vram_usage = float(os.getenv("MAX_VRAM_USAGE", "0.8"))
    model_manager = ModelManager(device=device, max_vram_usage=max_vram_usage)

    # Load default model (MuseTalk for best quality)
    available_models = model_manager.get_available_models()
    if "musetalk" in available_models:
        logger.info("Loading MuseTalk model...")
        success = model_manager.load_model("musetalk")
        if success:
            logger.info("✅ MuseTalk model loaded successfully")
        else:
            logger.warning("⚠️ Failed to load MuseTalk, trying alternatives...")
            # Try other models
            for model_id in ["wav2lip", "sadtalker"]:
                if model_id in available_models:
                    if model_manager.load_model(model_id):
                        logger.info(f"✅ {model_id} model loaded successfully")
                        break
    else:
        logger.warning("⚠️ No suitable models available")

    logger.info("🎉 Application startup complete!")

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("Shutting down application...")

    # Cleanup GPU memory
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Main application page"""
    # Get system stats
    stats = {}
    if model_manager:
        stats = model_manager.get_system_stats()

    # Get available models
    available_models = {}
    if model_manager:
        available_models = model_manager.get_available_models()

    # Get available voices
    available_voices = {}
    if tts_pipeline:
        available_voices = tts_pipeline.get_available_voices()

    return templates.TemplateResponse("index.html", {
        "request": request,
        "stats": stats,
        "available_models": available_models,
        "available_voices": available_voices
    })

@app.get("/api/status")
async def get_status():
    """Get application status and statistics"""
    status = {
        "status": "running",
        "timestamp": time.time(),
        "models": {},
        "tts": {},
        "system": {}
    }

    if model_manager:
        status["models"] = {
            "current_model": model_manager.current_model,
            "available_models": model_manager.get_available_models(),
            "system_stats": model_manager.get_system_stats()
        }

    if tts_pipeline:
        status["tts"] = {
            "available_voices": tts_pipeline.get_available_voices(),
            "stats": tts_pipeline.get_stats()
        }

    return status

@app.post("/api/models/load")
async def load_model(model_id: str = Form(...)):
    """Load a specific AI model"""
    if not model_manager:
        raise HTTPException(status_code=500, detail="Model manager not initialized")

    logger.info(f"Loading model: {model_id}")
    success = model_manager.load_model(model_id)

    if success:
        return {"status": "success", "message": f"Model {model_id} loaded successfully"}
    else:
        raise HTTPException(status_code=400, detail=f"Failed to load model {model_id}")

@app.post("/api/generate")
async def generate_video(
    background_tasks: BackgroundTasks,
    image: UploadFile = File(...),
    text: str = Form(...),
    voice: str = Form("Rachel"),
    model_id: str = Form(None),
    provider: str = Form(None)
):
    """Generate lipsync video from image and text"""
    if not model_manager or not tts_pipeline:
        raise HTTPException(status_code=500, detail="Services not initialized")

    # Validate file size
    if image.size > MAX_UPLOAD_SIZE:
        raise HTTPException(status_code=413, detail="File too large")

    # Generate unique session ID
    session_id = str(uuid.uuid4())

    try:
        # Save uploaded image
        image_path = UPLOAD_DIR / "images" / f"{session_id}_{image.filename}"
        with open(image_path, "wb") as f:
            content = await image.read()
            f.write(content)

        logger.info(f"Processing request {session_id}: {text[:50]}...")

        # Generate audio from text
        logger.info(f"Generating audio for session {session_id}")
        audio_path = await tts_pipeline.generate_audio(
            text=text,
            voice=voice,
            provider=provider,
            output_path=str(UPLOAD_DIR / "audio" / f"{session_id}.wav")
        )

        if not audio_path:
            raise HTTPException(status_code=500, detail="Failed to generate audio")

        # Load specific model if requested
        if model_id and model_id != model_manager.current_model:
            success = model_manager.load_model(model_id)
            if not success:
                raise HTTPException(status_code=400, detail=f"Failed to load model {model_id}")

        # Generate video
        logger.info(f"Generating video for session {session_id}")
        video_path = model_manager.generate_video(
            image_path=str(image_path),
            audio_path=audio_path
        )

        if not video_path:
            raise HTTPException(status_code=500, detail="Failed to generate video")

        # Return video file
        return FileResponse(
            video_path,
            media_type="video/mp4",
            filename=f"lipsync_{session_id}.mp4"
        )

    except Exception as e:
        logger.error(f"Video generation failed for session {session_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/generate-audio")
async def generate_audio_only(
    text: str = Form(...),
    voice: str = Form("Rachel"),
    provider: str = Form(None)
):
    """Generate audio from text only"""
    if not tts_pipeline:
        raise HTTPException(status_code=500, detail="TTS pipeline not initialized")

    session_id = str(uuid.uuid4())

    try:
        logger.info(f"Generating audio for session {session_id}: {text[:50]}...")

        audio_path = await tts_pipeline.generate_audio(
            text=text,
            voice=voice,
            provider=provider,
            output_path=str(UPLOAD_DIR / "audio" / f"{session_id}.wav")
        )

        if not audio_path:
            raise HTTPException(status_code=500, detail="Failed to generate audio")

        return FileResponse(
            audio_path,
            media_type="audio/wav",
            filename=f"tts_{session_id}.wav"
        )

    except Exception as e:
        logger.error(f"Audio generation failed for session {session_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/voices")
async def get_voices():
    """Get available TTS voices"""
    if not tts_pipeline:
        raise HTTPException(status_code=500, detail="TTS pipeline not initialized")

    return tts_pipeline.get_available_voices()

@app.get("/api/models")
async def get_models():
    """Get available AI models"""
    if not model_manager:
        raise HTTPException(status_code=500, detail="Model manager not initialized")

    return model_manager.get_available_models()

@app.delete("/api/cache/clear")
async def clear_cache():
    """Clear TTS cache"""
    if not tts_pipeline:
        raise HTTPException(status_code=500, detail="TTS pipeline not initialized")

    tts_pipeline.clear_cache()
    return {"status": "success", "message": "Cache cleared"}

if __name__ == "__main__":
    # Configuration from environment
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    workers = int(os.getenv("WORKERS", "1"))
    debug = os.getenv("DEBUG", "False").lower() == "true"

    logger.info(f"Starting server on {host}:{port}")

    uvicorn.run(
        "main:app",
        host=host,
        port=port,
        workers=workers,
        reload=debug,
        log_level="info"
    )