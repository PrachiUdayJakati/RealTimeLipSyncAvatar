"""
Model Manager for Real-Time Lipsync Avatar
Handles multiple AI models optimized for 16GB GPU VRAM
"""

import torch
import logging
import psutil
import os
from typing import Dict, Any, Optional, Tuple
from pathlib import Path
import gc

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelManager:
    """
    Manages multiple lipsync AI models with GPU memory optimization
    Supports: MuseTalk, SadTalker, Wav2Lip, and custom models
    """

    def __init__(self, device: str = "auto", max_vram_usage: float = 0.8):
        """
        Initialize model manager

        Args:
            device: Device to use ('auto', 'cuda', 'cpu')
            max_vram_usage: Maximum VRAM usage ratio (0.8 = 80% of 16GB = 12.8GB)
        """
        self.device = self._setup_device(device)
        self.max_vram_usage = max_vram_usage
        self.models = {}
        self.current_model = None
        self.model_configs = self._get_model_configs()

        logger.info(f"ModelManager initialized on {self.device}")
        logger.info(f"Max VRAM usage: {max_vram_usage * 100}%")

    def _setup_device(self, device: str) -> torch.device:
        """Setup and validate device"""
        if device == "auto":
            if torch.cuda.is_available():
                device = "cuda"
                gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
                logger.info(f"CUDA available - GPU Memory: {gpu_memory:.1f}GB")
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
                "vram_usage": 4.5,  # GB
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
            total_vram = torch.cuda.get_device_properties(0).total_memory / 1e9
            allocated_vram = torch.cuda.memory_allocated(0) / 1e9
            available_vram = (total_vram * self.max_vram_usage) - allocated_vram
            return max(0, available_vram)
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

    def generate_video(self, image_path: str, audio_path: str, **kwargs) -> Optional[str]:
        """
        Generate lipsync video using current model

        Args:
            image_path: Path to input image
            audio_path: Path to input audio
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
        # This will be implemented with actual MuseTalk loading
        logger.info("Loading MuseTalk model...")
        # Placeholder for actual implementation
        return {"type": "musetalk", "loaded": True}

    def _load_sadtalker(self):
        """Load SadTalker model"""
        # This will be implemented with actual SadTalker loading
        logger.info("Loading SadTalker model...")
        # Placeholder for actual implementation
        return {"type": "sadtalker", "loaded": True}

    def _load_wav2lip(self):
        """Load Wav2Lip model"""
        # This will be implemented with actual Wav2Lip loading
        logger.info("Loading Wav2Lip model...")
        # Placeholder for actual implementation
        return {"type": "wav2lip", "loaded": True}

    def _generate_musetalk(self, model, image_path: str, audio_path: str, **kwargs) -> str:
        """Generate video using MuseTalk"""
        # Placeholder for actual MuseTalk generation
        logger.info("Generating video with MuseTalk...")
        return "outputs/musetalk_output.mp4"

    def _generate_sadtalker(self, model, image_path: str, audio_path: str, **kwargs) -> str:
        """Generate video using SadTalker"""
        # Placeholder for actual SadTalker generation
        logger.info("Generating video with SadTalker...")
        return "outputs/sadtalker_output.mp4"

    def _generate_wav2lip(self, model, image_path: str, audio_path: str, **kwargs) -> str:
        """Generate video using Wav2Lip"""
        # Placeholder for actual Wav2Lip generation
        logger.info("Generating video with Wav2Lip...")
        return "outputs/wav2lip_output.mp4"