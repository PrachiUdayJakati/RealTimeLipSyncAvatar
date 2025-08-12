"""
Patched ModelManager with MuseTalk integration guidance and realistic fallback using MouthAnimator.
This file replaces the previous models/model_manager.py content with a version that:
- Tries to import a real MuseTalkModel from models.musetalk_model (if you have it)
- If MuseTalkModel is unavailable, uses a high-quality fallback that
  applies audio-driven mouth animation per-frame using mouth_animator.MouthAnimator
- Integrates with Bark-only workflow by expecting audio_path produced by TTSPipeline (Bark)
- Keeps VRAM-aware loading/unloading logic intact
- Adds clear logging and error handling
"""

import torch
import logging
import psutil
import os
import numpy as np
from typing import Dict, Any, Optional, Tuple
from pathlib import Path
import gc
import librosa
import cv2

# Import mouth animator we created earlier
try:
    from mouth_animator import MouthAnimator, MediapipeFaceMesh, MP_AVAILABLE
except Exception:
    MouthAnimator = None
    MediapipeFaceMesh = None
    MP_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelManager:
    """
    Manages multiple lipsync AI models with GPU memory optimization
    Supports: MuseTalk (preferred), SadTalker, Wav2Lip, and custom models
    If MuseTalk is not installed, falls back to a realistic audio-driven animator using MouthAnimator.
    """

    def __init__(self, device: str = "auto", max_vram_usage: float = 0.8, animator: Optional[MouthAnimator] = None):
        """
        Initialize model manager

        Args:
            device: Device to use ('auto', 'cuda', 'cpu')
            max_vram_usage: Maximum VRAM usage ratio (0.8 = 80% of 16GB = 12.8GB)
            animator: optional MouthAnimator instance (if not provided, one will be created if mediapipe available)
        """
        self.device = self._setup_device(device)
        self.max_vram_usage = max_vram_usage
        self.models = {}
        self.current_model = None
        self.model_configs = self._get_model_configs()

        # Hook in the mouth animator
        if animator is not None:
            self.animator = animator
        else:
            try:
                mp_inst = MediapipeFaceMesh(static_image_mode=False) if MP_AVAILABLE else None
                self.animator = MouthAnimator(use_landmarks=(mp_inst is not None), mediapipe_instance=mp_inst) if MouthAnimator is not None else None
            except Exception as e:
                logger.warning(f"Could not initialize MouthAnimator: {e}")
                self.animator = None

        logger.info(f"ModelManager initialized on {self.device}")
        logger.info(f"Max VRAM usage: {max_vram_usage * 100}%")

    def _setup_device(self, device: str) -> torch.device:
        """Setup and validate device"""
        if device == "auto":
            if torch.cuda.is_available():
                device = "cuda"
                try:
                    gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
                    logger.info(f"CUDA available - GPU Memory: {gpu_memory:.1f}GB")
                except Exception:
                    logger.info("CUDA available")
            else:
                device = "cpu"
                logger.warning("CUDA not available, using CPU")

        return torch.device(device)

    def _get_model_configs(self) -> Dict[str, Dict[str, Any]]:
        """Get configurations for different models"""
        return {
            "musetalk": {
                "name": "MuseTalk",
                "description": "Real-time high-quality lip synchronization",
                "vram_usage": 4.5,  # GB (estimate)
                "quality": "high",
                "speed": "fast",
                "repo_id": "TMElyralab/MuseTalk",
                "model_files": ["musetalk_model.pth", "face_encoder.pth"]
            },
            "sadtalker": {
                "name": "SadTalker",
                "description": "High-quality talking head generation",
                "vram_usage": 6.0,  # GB
                "quality": "very_high",
                "speed": "medium",
                "repo_id": "Winfredy/SadTalker",
                "model_files": ["sadtalker_model.pth", "face_3dmm.pth"]
            },
            "wav2lip": {
                "name": "Wav2Lip",
                "description": "Fast lip synchronization",
                "vram_usage": 2.5,  # GB
                "quality": "medium",
                "speed": "very_fast",
                "repo_id": "Rudrabha/Wav2Lip",
                "model_files": ["wav2lip_gan.pth"]
            }
        }

    def get_available_models(self) -> Dict[str, Dict[str, Any]]:
        """Get list of available models with their specs"""
        available_vram = self._get_available_vram()

        available_models = {}
        for model_id, config in self.model_configs.items():
            if config["vram_usage"] <= available_vram:
                available_models[model_id] = config
            else:
                logger.warning(f"Model {model_id} requires {config['vram_usage']}GB VRAM, "
                             f"but only {available_vram:.1f}GB available")

        return available_models

    def _get_available_vram(self) -> float:
        """Get available VRAM in GB"""
        if self.device.type == "cuda":
            try:
                total_vram = torch.cuda.get_device_properties(0).total_memory / 1e9
                allocated_vram = torch.cuda.memory_allocated(0) / 1e9
                available_vram = (total_vram * self.max_vram_usage) - allocated_vram
                return max(0, available_vram)
            except Exception:
                return 0.0
        return float('inf')  # CPU has no VRAM limit

    def load_model(self, model_id: str, force_reload: bool = False) -> bool:
        """
        Load a specific model

        Args:
            model_id: Model identifier ('musetalk', 'sadtalker', 'wav2lip')
            force_reload: Force reload even if already loaded

        Returns:
            bool: Success status
        """
        if model_id not in self.model_configs:
            logger.error(f"Unknown model: {model_id}")
            return False

        if model_id in self.models and not force_reload:
            logger.info(f"Model {model_id} already loaded")
            self.current_model = model_id
            return True

        # Check VRAM availability
        required_vram = self.model_configs[model_id]["vram_usage"]
        available_vram = self._get_available_vram()

        if required_vram > available_vram:
            logger.error(f"Insufficient VRAM: need {required_vram}GB, have {available_vram:.1f}GB")
            return False

        # Unload current model if needed
        if self.current_model and self.current_model != model_id:
            self.unload_model(self.current_model)

        try:
            logger.info(f"Loading {model_id} model...")

            if model_id == "musetalk":
                model = self._load_musetalk()
            elif model_id == "sadtalker":
                model = self._load_sadtalker()
            elif model_id == "wav2lip":
                model = self._load_wav2lip()
            else:
                raise ValueError(f"Model loader not implemented for {model_id}")

            self.models[model_id] = model
            self.current_model = model_id

            logger.info(f"Successfully loaded {model_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to load {model_id}: {str(e)}")
            return False

    def unload_model(self, model_id: str) -> bool:
        """Unload a specific model to free VRAM"""
        if model_id not in self.models:
            logger.warning(f"Model {model_id} not loaded")
            return False

        try:
            del self.models[model_id]
            if self.current_model == model_id:
                self.current_model = None

            # Force garbage collection and clear CUDA cache
            gc.collect()
            if self.device.type == "cuda":
                torch.cuda.empty_cache()

            logger.info(f"Unloaded model {model_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to unload {model_id}: {str(e)}")
            return False

    def generate_video(self, image_path: str, audio_path: str, output_path: str = None, **kwargs) -> Optional[str]:
        """
        Generate lipsync video using current model - WRAPPER METHOD
        """
        logger.info(f"ModelManager.generate_video called with args: image_path={image_path}, audio_path={audio_path}, output_path={output_path}")
        return self._internal_generate_video(image_path, audio_path, output_path, **kwargs)

    def _internal_generate_video(self, image_path: str, audio_path: str, output_path: str = None, **kwargs) -> Optional[str]:
        """
        Generate lipsync video using current model

        Args:
            image_path: Path to input image
            audio_path: Path to input audio (expected to be produced by Bark)
            **kwargs: Additional model-specific parameters

        Returns:
            str: Path to generated video or None if failed
        """
        if not self.current_model:
            logger.error("No model loaded")
            return None

        if self.current_model not in self.models:
            logger.error(f"Current model {self.current_model} not found")
            return None

        try:
            model = self.models[self.current_model]

            if self.current_model == "musetalk":
                return self._generate_musetalk(model, image_path, audio_path, **kwargs)
            elif self.current_model == "sadtalker":
                return self._generate_sadtalker(model, image_path, audio_path, **kwargs)
            elif self.current_model == "wav2lip":
                return self._generate_wav2lip(model, image_path, audio_path, **kwargs)
            else:
                raise ValueError(f"Generator not implemented for {self.current_model}")

        except Exception as e:
            logger.error(f"Video generation failed: {str(e)}")
            return None

    def get_system_stats(self) -> Dict[str, Any]:
        """Get current system and GPU statistics"""
        stats = {
            "device": str(self.device),
            "current_model": self.current_model,
            "loaded_models": list(self.models.keys()),
            "cpu_usage": psutil.cpu_percent(),
            "ram_usage": psutil.virtual_memory().percent,
            "ram_available": psutil.virtual_memory().available / 1e9,  # GB
        }

        if self.device.type == "cuda":
            stats.update({
                "gpu_name": torch.cuda.get_device_name(0),
                "total_vram": torch.cuda.get_device_properties(0).total_memory / 1e9,
                "allocated_vram": torch.cuda.memory_allocated(0) / 1e9,
                "cached_vram": torch.cuda.memory_reserved(0) / 1e9,
                "available_vram": self._get_available_vram()
            })

        return stats

    def _load_musetalk(self):
        """Load MuseTalk model"""
        logger.info("Loading MuseTalk model...")
        # Try to import a local adapter that wraps the official MuseTalk repo
        try:
            from models.musetalk_model import MuseTalkModel  # user-provided adapter
            mt = MuseTalkModel(device=self.device)
            mt.load_models()
            logger.info("MuseTalkModel adapter loaded successfully")
            return {"type": "musetalk", "instance": mt}
        except Exception as e:
            logger.warning(f"MuseTalkModel adapter not found or failed to load: {e}")
            # As a fallback, register a lightweight placeholder that will NOT crash but will use animator fallback
            return {"type": "musetalk", "instance": None, "fallback": True}

    def _load_sadtalker(self):
        """Load SadTalker model (placeholder)"""
        logger.info("Loading SadTalker model (placeholder)...")
        # User can implement actual loader here
        return {"type": "sadtalker", "loaded": False}

    def _load_wav2lip(self):
        """Load Wav2Lip model (placeholder)"""
        logger.info("Loading Wav2Lip model (placeholder)...")
        # User can implement actual loader here
        return {"type": "wav2lip", "loaded": False}

    def _generate_musetalk(self, model, image_path: str, audio_path: str, **kwargs) -> str:
        """Generate video using MuseTalk or fallback to animator-driven video"""
        logger.info("Generating video with MuseTalk...")

        try:
            inst = None
            if isinstance(model, dict) and "instance" in model:
                inst = model["instance"]

            # If we have an actual MuseTalk instance, call its generate function
            if inst is not None:
                try:
                    output_path = inst.generate_video(image_path=image_path, audio_path=audio_path, **kwargs)
                    if output_path and os.path.exists(output_path):
                        logger.info(f"MuseTalk generated video: {output_path}")
                        return output_path
                    else:
                        raise Exception("MuseTalk adapter returned invalid path")
                except Exception as e:
                    logger.exception(f"MuseTalk adapter failed: {e}. Falling back to animator-based generator.")

            # Fallback: use animator-driven frame-by-frame generator (real-time friendly)
            return self._create_animator_video(image_path, audio_path, output_path, **kwargs)

        except Exception as e:
            logger.error(f"MuseTalk generation failed entirely: {e}")
            return self._create_fallback_video(image_path, audio_path)

    def _generate_sadtalker(self, model, image_path: str, audio_path: str, **kwargs) -> str:
        """Generate video using SadTalker (placeholder)"""
        logger.info("Generating video with SadTalker (placeholder)...")
        return "outputs/sadtalker_output.mp4"

    def _generate_wav2lip(self, model, image_path: str, audio_path: str, **kwargs) -> str:
        """Generate video using Wav2Lip (placeholder)"""
        logger.info("Generating video with Wav2Lip (placeholder)...")
        return "outputs/wav2lip_output.mp4"

    def _create_animator_video(self, image_path: str, audio_path: str, output_path: Optional[str] = None, fps: int = 25, **kwargs) -> str:
        """
        Create a higher-quality fallback video using the MouthAnimator (if available).
        This replaces the old dark-mouth fallback with audio-driven, texture-preserving deformation.
        """
        try:
            if self.animator is None:
                logger.warning("Animator not available; using old fallback video creation")
                return self._create_fallback_video(image_path, audio_path)

            # Load base image
            img = cv2.imread(image_path)
            if img is None:
                raise ValueError(f"Could not load image: {image_path}")

            # Resize to a square for stable output
            target_size = kwargs.get("size", (512, 512))
            img = cv2.resize(img, target_size)

            # Load audio (mono)
            audio, sr = librosa.load(audio_path, sr=None)
            if sr != 16000:
                audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
                sr = 16000

            duration = len(audio) / sr
            num_frames = max(int(duration * fps), 1)

            # Ensure outputs folder exists
            os.makedirs("outputs", exist_ok=True)
            temp_video = "outputs/temp_animator_video.mp4"
            final_video = output_path or "outputs/musetalk_animator_output.mp4"

            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(temp_video, fourcc, fps, target_size)

            # Chunk audio into per-frame windows and feed to animator.ingest_audio
            samples_per_frame = int(sr / fps)
            half_win = max(1, samples_per_frame // 2)

            # Optionally get a face bbox from kwargs or use center heuristics
            face_bbox = kwargs.get("face_bbox", (int(target_size[0]*0.2), int(target_size[1]*0.12),
                                                int(target_size[0]*0.8), int(target_size[1]*0.9)))

            for i in range(num_frames):
                # compute audio window centered at the frame time
                center = i * samples_per_frame
                start = max(0, center - half_win)
                end = min(len(audio), center + half_win)
                chunk = audio[start:end]
                # ingest chunk (updates animator's internal EMA)
                self.animator.ingest_audio(chunk)
                # apply to frame
                frame = self.animator.apply_to_frame(img.copy(), face_bbox, audio_features=None, frame_idx=i)
                out.write(frame)

            out.release()

            # Merge audio using ffmpeg to keep quality
            cmd = [
                'ffmpeg', '-y',
                '-i', temp_video,
                '-i', audio_path,
                '-c:v', 'libx264',
                '-c:a', 'aac',
                '-shortest',
                '-pix_fmt', 'yuv420p',
                final_video
            ]

            import subprocess
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                logger.warning(f"ffmpeg merge failed: {result.stderr}; returning temp video path")
                return temp_video

            # cleanup temp
            if os.path.exists(temp_video):
                os.remove(temp_video)

            logger.info(f"Animator-based video created: {final_video}")
            return final_video

        except Exception as e:
            logger.exception(f"Animator video creation failed: {e}")
            return self._create_fallback_video(image_path, audio_path)

    def _create_fallback_video(self, image_path: str, audio_path: str) -> str:
        """Create a fallback video with basic lip sync animation and audio (less realistic)."""
        import cv2
        import numpy as np
        import librosa
        import subprocess
        import os

        try:
            logger.info("Creating fallback video with audio (legacy method)...")

            # Load image
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Could not load image: {image_path}")

            # Resize image
            image = cv2.resize(image, (512, 512))

            # Get audio duration
            audio, sr = librosa.load(audio_path, sr=None)
            duration = len(audio) / sr

            # Create video parameters
            fps = 25
            num_frames = max(int(duration * fps), 75)  # At least 3 seconds

            # Output paths
            temp_video = "outputs/temp_video.mp4"
            final_video = "outputs/musetalk_output.mp4"
            os.makedirs("outputs", exist_ok=True)

            # Create video with basic mouth animation
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(temp_video, fourcc, fps, (512, 512))

            for frame_idx in range(num_frames):
                # Create animated frame
                frame = image.copy()

                # Simple mouth animation (darken mouth area periodically)
                mouth_open = abs(np.sin(frame_idx * 0.3)) * 0.4 + 0.1

                # Mouth region (approximate)
                h, w = frame.shape[:2]
                mouth_y1, mouth_y2 = int(h * 0.65), int(h * 0.85)
                mouth_x1, mouth_x2 = int(w * 0.35), int(w * 0.65)

                # Apply mouth animation
                mouth_region = frame[mouth_y1:mouth_y2, mouth_x1:mouth_x2].copy()
                darkness = 1.0 - (mouth_open * 0.5)
                mouth_region = (mouth_region * darkness).astype(np.uint8)
                frame[mouth_y1:mouth_y2, mouth_x1:mouth_x2] = mouth_region

                out.write(frame)

            out.release()

            # Merge with audio using FFmpeg
            cmd = [
                'ffmpeg', '-y',
                '-i', temp_video,
                '-i', audio_path,
                '-c:v', 'libx264',
                '-c:a', 'aac',
                '-shortest',
                '-pix_fmt', 'yuv420p',
                final_video
            ]

            result = subprocess.run(cmd, capture_output=True, text=True)

            if result.returncode == 0:
                # Clean up temp file
                if os.path.exists(temp_video):
                    os.remove(temp_video)
                logger.info(f"Fallback video with audio created: {final_video}")
                return final_video
            else:
                logger.error(f"FFmpeg failed: {result.stderr}")
                # Return video without audio as last resort
                if os.path.exists(temp_video):
                    os.rename(temp_video, final_video)
                return final_video

        except Exception as e:
            logger.error(f"Fallback video creation failed: {e}")
            # Create minimal video
            return self._create_minimal_video(image_path)

    def _create_minimal_video(self, image_path: str) -> str:
        """Create minimal video as last resort"""
        import cv2
        import os

        try:
            image = cv2.imread(image_path)
            if image is None:
                # Create a black frame
                image = np.zeros((512, 512, 3), dtype=np.uint8)

            image = cv2.resize(image, (512, 512))

            output_path = "outputs/musetalk_output.mp4"
            os.makedirs("outputs", exist_ok=True)

            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, 25, (512, 512))

            # Write 75 frames (3 seconds)
            for _ in range(75):
                out.write(image)

            out.release()
            return output_path

        except Exception as e:
            logger.error(f"Minimal video creation failed: {e}")
            return "outputs/musetalk_output.mp4"
