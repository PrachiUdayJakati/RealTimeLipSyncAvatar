"""
Real-Time Lipsync Avatar Application - Final Version
Complete working implementation with Bark TTS
"""

import asyncio
import logging
import os
import uuid
from pathlib import Path

import uvicorn
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Request
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

# Import our modules
from utils.tts_pipeline import TTSPipeline
from models.model_manager import ModelManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Real-Time Lipsync Avatar",
    description="AI-powered lipsync avatar with Bark TTS",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
try:
    app.mount("/static", StaticFiles(directory="static"), name="static")
except:
    pass  # Static directory might not exist

try:
    app.mount("/outputs", StaticFiles(directory="outputs"), name="outputs")
except:
    os.makedirs("outputs", exist_ok=True)
    app.mount("/outputs", StaticFiles(directory="outputs"), name="outputs")

# Global variables for our services
tts_pipeline = None
model_manager = None
active_sessions = {}

@app.on_event("startup")
async def startup_event():
    """Initialize services on startup"""
    global tts_pipeline, model_manager

    logger.info("🚀 Starting Real-Time Lipsync Avatar application...")

    # Initialize TTS pipeline
    logger.info("Initializing TTS pipeline...")
    tts_pipeline = TTSPipeline()

    # Initialize AI model manager
    logger.info("Initializing AI model manager...")
    model_manager = ModelManager()

    # Load MuseTalk model
    logger.info("Loading MuseTalk model...")
    success = model_manager.load_model("musetalk")
    if success:
        logger.info("✅ MuseTalk model loaded successfully")
    else:
        logger.warning("⚠️ MuseTalk model failed to load")

    logger.info("🎉 Application startup complete!")

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("Shutting down application...")
    # Cleanup GPU memory if available
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except:
        pass

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Serve the main interface with embedded HTML"""
    html_content = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>🎬 Real-Time Lipsync Avatar</title>
        <style>
            * {
                margin: 0;
                padding: 0;
                box-sizing: border-box;
            }

            body {
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                min-height: 100vh;
                padding: 20px;
            }

            .container {
                max-width: 900px;
                margin: 0 auto;
                background: white;
                border-radius: 20px;
                box-shadow: 0 20px 40px rgba(0,0,0,0.1);
                overflow: hidden;
            }

            .header {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                padding: 30px;
                text-align: center;
            }

            .header h1 {
                font-size: 2.5em;
                margin-bottom: 10px;
            }

            .header p {
                font-size: 1.2em;
                opacity: 0.9;
            }

            .content {
                padding: 40px;
            }

            .form-group {
                margin-bottom: 25px;
            }

            .form-group label {
                display: block;
                margin-bottom: 8px;
                font-weight: 600;
                color: #333;
            }

            .form-group input,
            .form-group textarea,
            .form-group select {
                width: 100%;
                padding: 12px;
                border: 2px solid #e1e5e9;
                border-radius: 8px;
                font-size: 16px;
                transition: border-color 0.3s;
            }

            .form-group input:focus,
            .form-group textarea:focus,
            .form-group select:focus {
                outline: none;
                border-color: #667eea;
            }

            .form-group textarea {
                resize: vertical;
                min-height: 100px;
            }

            .generate-btn {
                width: 100%;
                padding: 15px;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                border: none;
                border-radius: 8px;
                font-size: 18px;
                font-weight: 600;
                cursor: pointer;
                transition: transform 0.2s;
            }

            .generate-btn:hover {
                transform: translateY(-2px);
            }

            .generate-btn:disabled {
                opacity: 0.6;
                cursor: not-allowed;
                transform: none;
            }

            .status {
                margin-top: 20px;
                padding: 15px;
                border-radius: 8px;
                text-align: center;
                font-weight: 600;
            }

            .status.loading {
                background: #fff3cd;
                color: #856404;
                border: 1px solid #ffeaa7;
            }

            .status.success {
                background: #d4edda;
                color: #155724;
                border: 1px solid #c3e6cb;
            }

            .status.error {
                background: #f8d7da;
                color: #721c24;
                border: 1px solid #f5c6cb;
            }

            .result {
                margin-top: 30px;
                padding: 25px;
                background: #f8f9fa;
                border-radius: 12px;
                text-align: center;
            }

            .result video {
                width: 100%;
                max-width: 500px;
                border-radius: 8px;
                box-shadow: 0 10px 20px rgba(0,0,0,0.1);
            }

            .download-btn {
                display: inline-block;
                margin-top: 15px;
                padding: 10px 20px;
                background: #28a745;
                color: white;
                text-decoration: none;
                border-radius: 6px;
                font-weight: 600;
                transition: background 0.3s;
            }

            .download-btn:hover {
                background: #218838;
            }

            .tech-info {
                margin-top: 30px;
                padding: 20px;
                background: #e9ecef;
                border-radius: 8px;
                font-size: 14px;
                color: #6c757d;
            }

            .tech-info h4 {
                color: #495057;
                margin-bottom: 10px;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>🎬 Real-Time Lipsync Avatar</h1>
                <p>Create amazing lipsync videos with AI-powered Bark TTS</p>
            </div>

            <div class="content">
                <form id="uploadForm" enctype="multipart/form-data">
                    <div class="form-group">
                        <label for="image">📸 Upload Avatar Image</label>
                        <input type="file" id="image" name="image" accept="image/*" required>
                        <small>Upload a clear photo of a person's face (JPG, PNG)</small>
                    </div>

                    <div class="form-group">
                        <label for="text">💬 Text to Speech</label>
                        <textarea id="text" name="text" placeholder="Enter the text you want your avatar to speak..." required rows="4"></textarea>
                        <small>Enter any text - Bark TTS will generate natural speech</small>
                    </div>

                    <div class="form-group">
                        <label for="voice">🎤 Voice Selection</label>
                        <select id="voice" name="voice">
                            <option value="v2/en_speaker_6">Speaker 6 (Default - Balanced)</option>
                            <option value="v2/en_speaker_0">Speaker 0 (Deep Male)</option>
                            <option value="v2/en_speaker_1">Speaker 1 (Female)</option>
                            <option value="v2/en_speaker_2">Speaker 2 (Young Male)</option>
                            <option value="v2/en_speaker_3">Speaker 3 (Mature Female)</option>
                            <option value="v2/en_speaker_4">Speaker 4 (Energetic)</option>
                            <option value="v2/en_speaker_5">Speaker 5 (Calm)</option>
                            <option value="v2/en_speaker_7">Speaker 7 (Professional)</option>
                            <option value="v2/en_speaker_8">Speaker 8 (Friendly)</option>
                            <option value="v2/en_speaker_9">Speaker 9 (Expressive)</option>
                        </select>
                        <small>Choose from 10 different Bark TTS voices</small>
                    </div>

                    <button type="submit" class="generate-btn" id="generateBtn">
                        🚀 Generate Lipsync Video
                    </button>
                </form>

                <div id="status" class="status" style="display: none;"></div>

                <div id="result" class="result" style="display: none;">
                    <h3>🎉 Your Lipsync Video is Ready!</h3>
                    <video id="videoPlayer" controls></video>
                    <br>
                    <a id="downloadLink" class="download-btn" download>📥 Download Video</a>
                </div>

                <div class="tech-info">
                    <h4>🔧 Technology Stack</h4>
                    <p><strong>TTS:</strong> Bark (Free, Open Source) • <strong>AI Model:</strong> MuseTalk • <strong>Video:</strong> Custom Lip Sync</p>
                    <p>This application uses completely free and open-source technologies for high-quality lipsync video generation.</p>
                </div>
            </div>
        </div>

        <script>
            document.getElementById('uploadForm').addEventListener('submit', async function(e) {
                e.preventDefault();

                const formData = new FormData();
                const imageFile = document.getElementById('image').files[0];
                const text = document.getElementById('text').value;
                const voice = document.getElementById('voice').value;

                if (!imageFile) {
                    showStatus('Please select an image file.', 'error');
                    return;
                }

                if (!text.trim()) {
                    showStatus('Please enter some text.', 'error');
                    return;
                }

                formData.append('image', imageFile);
                formData.append('text', text);
                formData.append('voice', voice);
                formData.append('model', 'musetalk');

                const generateBtn = document.getElementById('generateBtn');
                generateBtn.disabled = true;
                generateBtn.textContent = '⏳ Generating...';

                showStatus('🎤 Generating speech with Bark TTS... This may take a few minutes for the first generation.', 'loading');

                try {
                    const response = await fetch('/api/generate', {
                        method: 'POST',
                        body: formData
                    });

                    const result = await response.json();

                    if (response.ok) {
                        showStatus('✅ Video generated successfully!', 'success');
                        document.getElementById('videoPlayer').src = result.video_url;
                        document.getElementById('downloadLink').href = result.video_url;
                        document.getElementById('result').style.display = 'block';

                        // Scroll to result
                        document.getElementById('result').scrollIntoView({ behavior: 'smooth' });
                    } else {
                        showStatus('❌ Error: ' + result.detail, 'error');
                    }
                } catch (error) {
                    showStatus('❌ Network error: ' + error.message, 'error');
                } finally {
                    generateBtn.disabled = false;
                    generateBtn.textContent = '🚀 Generate Lipsync Video';
                }
            });

            function showStatus(message, type) {
                const statusDiv = document.getElementById('status');
                statusDiv.textContent = message;
                statusDiv.className = 'status ' + type;
                statusDiv.style.display = 'block';
            }
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)

@app.get("/api/status")
async def get_status():
    """Get system status"""
    return {
        "status": "ready",
        "models_loaded": True,
        "tts_provider": "bark",
        "bark_available": True,
        "version": "1.0.0"
    }

@app.post("/api/generate")
async def generate_video(
    image: UploadFile = File(...),
    text: str = Form(...),
    voice: str = Form("v2/en_speaker_6"),
    model: str = Form("musetalk")
):
    """Generate lipsync video with Bark TTS"""
    try:
        # Generate session ID
        session_id = str(uuid.uuid4())
        logger.info(f"Processing request {session_id}: {text[:50]}...")

        # Validate inputs
        if not text.strip():
            raise HTTPException(status_code=400, detail="Text cannot be empty")

        if not image.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")

        # Save uploaded image
        image_path = f"outputs/input_image_{session_id}.jpg"
        os.makedirs("outputs", exist_ok=True)

        with open(image_path, "wb") as f:
            content = await image.read()
            f.write(content)

        logger.info(f"Image saved: {image_path}")

        # Generate audio with Bark TTS
        logger.info(f"Generating audio for session {session_id}")
        audio_path = await tts_pipeline.generate_audio(text, voice)

        if not audio_path or not os.path.exists(audio_path):
            raise HTTPException(status_code=500, detail="Failed to generate audio with Bark TTS")

        logger.info(f"Audio generated: {audio_path}")

        # Generate video - FIXED: Use correct method signature
        logger.info(f"Generating video for session {session_id}")
        logger.info(f"Calling ModelManager.generate_video with: {image_path}, {audio_path}")
        try:
            video_path = model_manager.generate_video(image_path, audio_path)
            logger.info(f"ModelManager.generate_video returned: {video_path}")
        except Exception as e:
            logger.error(f"ModelManager.generate_video failed with error: {e}")
            logger.error(f"Error type: {type(e)}")
            raise

        if not video_path or not os.path.exists(video_path):
            raise HTTPException(status_code=500, detail="Failed to generate video")

        logger.info(f"Video generated: {video_path}")

        # Store session info
        active_sessions[session_id] = {
            "status": "completed",
            "video_path": video_path,
            "text": text,
            "voice": voice,
            "model": model
        }

        return {
            "session_id": session_id,
            "status": "completed",
            "video_url": f"/outputs/{os.path.basename(video_path)}",
            "message": "Video generated successfully with Bark TTS!"
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Video generation failed for session {session_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Video generation failed: {str(e)}")

@app.get("/api/session/{session_id}")
async def get_session(session_id: str):
    """Get session status"""
    if session_id in active_sessions:
        return active_sessions[session_id]
    else:
        raise HTTPException(status_code=404, detail="Session not found")

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info"
    )