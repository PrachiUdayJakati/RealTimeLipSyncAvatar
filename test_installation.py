#!/usr/bin/env python3
"""
Installation Test Script for Real-Time Lipsync Avatar
Tests all major components and dependencies
"""

import sys
import os
import logging
import traceback
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def test_python_version():
    """Test Python version compatibility"""
    logger.info("Testing Python version...")
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        logger.error(f"Python 3.8+ required, found {version.major}.{version.minor}")
        return False
    logger.info(f"✅ Python {version.major}.{version.minor}.{version.micro}")
    return True

def test_pytorch():
    """Test PyTorch installation and CUDA support"""
    logger.info("Testing PyTorch...")
    try:
        import torch
        logger.info(f"✅ PyTorch {torch.__version__}")

        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
            logger.info(f"✅ CUDA available - {gpu_count} GPU(s)")
            logger.info(f"   GPU: {gpu_name}")
            logger.info(f"   VRAM: {gpu_memory:.1f}GB")

            if gpu_memory < 4:
                logger.warning("⚠️  Low VRAM detected. Consider using Wav2Lip model only.")

            return True
        else:
            logger.warning("⚠️  CUDA not available - will use CPU (slower)")
            return True

    except ImportError as e:
        logger.error(f"❌ PyTorch not found: {e}")
        return False

def test_core_dependencies():
    """Test core dependencies"""
    logger.info("Testing core dependencies...")

    dependencies = [
        ('fastapi', 'FastAPI'),
        ('uvicorn', 'Uvicorn'),
        ('opencv-python', 'cv2'),
        ('numpy', 'numpy'),
        ('librosa', 'librosa'),
        ('PIL', 'Pillow'),
        ('transformers', 'transformers'),
        ('diffusers', 'diffusers')
    ]

    failed = []
    for package, import_name in dependencies:
        try:
            __import__(import_name)
            logger.info(f"✅ {package}")
        except ImportError:
            logger.error(f"❌ {package} not found")
            failed.append(package)

    return len(failed) == 0

def test_optional_dependencies():
    """Test optional dependencies"""
    logger.info("Testing optional dependencies...")

    optional_deps = [
        ('elevenlabs', 'ElevenLabs TTS'),
        ('edge_tts', 'Edge TTS'),
        ('insightface', 'InsightFace'),
        ('onnxruntime', 'ONNX Runtime')
    ]

    for package, description in optional_deps:
        try:
            __import__(package)
            logger.info(f"✅ {description}")
        except ImportError:
            logger.warning(f"⚠️  {description} not available")

def test_model_manager():
    """Test model manager initialization"""
    logger.info("Testing model manager...")
    try:
        from models.model_manager import ModelManager

        device = "cuda" if torch.cuda.is_available() else "cpu"
        manager = ModelManager(device=device)

        available_models = manager.get_available_models()
        logger.info(f"✅ Model manager initialized")
        logger.info(f"   Available models: {list(available_models.keys())}")

        return True

    except Exception as e:
        logger.error(f"❌ Model manager failed: {e}")
        return False

def test_tts_pipeline():
    """Test TTS pipeline"""
    logger.info("Testing TTS pipeline...")
    try:
        from utils.tts_pipeline import TTSPipeline

        tts = TTSPipeline()
        voices = tts.get_available_voices()
        stats = tts.get_stats()

        logger.info(f"✅ TTS pipeline initialized")
        logger.info(f"   Available providers: {list(voices.keys())}")

        return True

    except Exception as e:
        logger.error(f"❌ TTS pipeline failed: {e}")
        return False

def test_directory_structure():
    """Test required directory structure"""
    logger.info("Testing directory structure...")

    required_dirs = [
        'models', 'utils', 'templates', 'static/css',
        'uploads/images', 'uploads/audio', 'outputs/videos',
        'cache/audio'
    ]

    missing = []
    for dir_path in required_dirs:
        if not Path(dir_path).exists():
            missing.append(dir_path)

    if missing:
        logger.warning(f"⚠️  Missing directories: {missing}")
        logger.info("Creating missing directories...")
        for dir_path in missing:
            Path(dir_path).mkdir(parents=True, exist_ok=True)
        logger.info("✅ Directories created")
    else:
        logger.info("✅ Directory structure complete")

    return True

def test_environment_config():
    """Test environment configuration"""
    logger.info("Testing environment configuration...")

    if Path('.env').exists():
        logger.info("✅ .env file found")

        # Check for API keys
        elevenlabs_key = os.getenv('ELEVENLABS_API_KEY')
        if elevenlabs_key and elevenlabs_key != 'your_elevenlabs_api_key_here':
            logger.info("✅ ElevenLabs API key configured")
        else:
            logger.warning("⚠️  ElevenLabs API key not configured")

    else:
        logger.warning("⚠️  .env file not found")
        logger.info("   Copy .env.template to .env and configure your API keys")

    return True

def main():
    """Run all tests"""
    logger.info("🚀 Starting Real-Time Lipsync Avatar installation test...")
    logger.info("=" * 60)

    tests = [
        test_python_version,
        test_pytorch,
        test_core_dependencies,
        test_optional_dependencies,
        test_directory_structure,
        test_environment_config,
        test_model_manager,
        test_tts_pipeline
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            if test():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            logger.error(f"❌ Test {test.__name__} crashed: {e}")
            logger.debug(traceback.format_exc())
            failed += 1

        logger.info("-" * 40)

    logger.info("=" * 60)
    logger.info(f"📊 Test Results: {passed} passed, {failed} failed")

    if failed == 0:
        logger.info("🎉 All tests passed! Your installation is ready.")
        logger.info("   Run 'python main.py' to start the application")
    else:
        logger.error("❌ Some tests failed. Please check the errors above.")
        logger.info("   Install missing dependencies and try again")

    return failed == 0

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)