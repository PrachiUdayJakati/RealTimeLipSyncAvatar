"""
Enhanced Text-to-Speech Pipeline for Real-Time Lipsync Avatar
Supports multiple TTS providers with fallback options
"""

import os
import logging
import tempfile
import time
from typing import Optional, Dict, Any, List
from pathlib import Path
import asyncio
import aiofiles
from pydub import AudioSegment
import librosa
import soundfile as sf
import numpy as np
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)

class TTSPipeline:
    """
    Advanced TTS pipeline with multiple providers and real-time optimization
    Optimized for low-latency audio generation
    """

    def __init__(self, cache_dir: str = "cache/audio"):
        """
        Initialize TTS pipeline

        Args:
            cache_dir: Directory for audio caching
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # TTS providers configuration
        self.providers = {
            "elevenlabs": {
                "enabled": self._check_elevenlabs(),
                "priority": 1,
                "quality": "high",
                "speed": "fast"
            },
            "edge_tts": {
                "enabled": self._check_edge_tts(),
                "priority": 2,
                "quality": "medium",
                "speed": "very_fast"
            },
            "espnet": {
                "enabled": self._check_espnet(),
                "priority": 3,
                "quality": "medium",
                "speed": "medium"
            }
        }

        # Audio configuration
        self.target_sample_rate = 16000
        self.target_channels = 1
        self.audio_format = "wav"

        # Cache settings
        self.enable_cache = True
        self.max_cache_size_mb = 500  # 500MB cache limit

        logger.info("TTS Pipeline initialized")
        self._log_available_providers()

    def _check_elevenlabs(self) -> bool:
        """Check if ElevenLabs is available"""
        try:
            api_key = os.getenv("ELEVENLABS_API_KEY")
            if not api_key:
                logger.warning("ELEVENLABS_API_KEY not found in environment")
                return False

            from elevenlabs import set_api_key, voices
            set_api_key(api_key)

            # Test API connection
            voices.get_voices()
            logger.info("ElevenLabs TTS available")
            return True

        except Exception as e:
            logger.warning(f"ElevenLabs not available: {str(e)}")
            return False

    def _check_edge_tts(self) -> bool:
        """Check if Edge TTS is available"""
        try:
            import edge_tts
            logger.info("Edge TTS available")
            return True
        except ImportError:
            logger.warning("Edge TTS not available (pip install edge-tts)")
            return False

    def _check_espnet(self) -> bool:
        """Check if ESPnet TTS is available"""
        try:
            import espnet2
            logger.info("ESPnet TTS available")
            return True
        except ImportError:
            logger.warning("ESPnet TTS not available")
            return False

    def _log_available_providers(self):
        """Log available TTS providers"""
        available = [name for name, config in self.providers.items() if config["enabled"]]
        if available:
            logger.info(f"Available TTS providers: {', '.join(available)}")
        else:
            logger.error("No TTS providers available!")

    def get_available_voices(self, provider: str = None) -> Dict[str, List[str]]:
        """Get available voices for each provider"""
        voices = {}

        for provider_name, config in self.providers.items():
            if provider and provider_name != provider:
                continue

            if not config["enabled"]:
                continue

            try:
                if provider_name == "elevenlabs":
                    voices[provider_name] = self._get_elevenlabs_voices()
                elif provider_name == "edge_tts":
                    voices[provider_name] = self._get_edge_voices()
                elif provider_name == "espnet":
                    voices[provider_name] = self._get_espnet_voices()
            except Exception as e:
                logger.error(f"Failed to get voices for {provider_name}: {str(e)}")

        return voices

    def _get_elevenlabs_voices(self) -> List[str]:
        """Get ElevenLabs voices"""
        try:
            from elevenlabs import voices
            voice_list = voices.get_voices()
            return [voice.name for voice in voice_list.voices]
        except:
            return ["Rachel", "Adam", "Domi", "Elli", "Josh", "Arnold", "Antoni", "Sam"]

    def _get_edge_voices(self) -> List[str]:
        """Get Edge TTS voices"""
        return [
            "en-US-AriaNeural",
            "en-US-JennyNeural",
            "en-US-GuyNeural",
            "en-US-AndrewNeural",
            "en-US-EmmaNeural",
            "en-US-BrianNeural"
        ]

    def _get_espnet_voices(self) -> List[str]:
        """Get ESPnet voices"""
        return ["ljspeech", "vctk", "libritts"]

    async def generate_audio(self, text: str, voice: str = None, provider: str = None,
                           output_path: str = None, **kwargs) -> Optional[str]:
        """
        Generate audio from text using the best available provider

        Args:
            text: Text to convert to speech
            voice: Voice to use (provider-specific)
            provider: Specific provider to use (optional)
            output_path: Output file path (optional)
            **kwargs: Additional provider-specific parameters

        Returns:
            str: Path to generated audio file or None if failed
        """
        start_time = time.time()

        # Check cache first
        if self.enable_cache:
            cached_path = self._check_cache(text, voice, provider)
            if cached_path:
                logger.info(f"Using cached audio: {cached_path}")
                return cached_path

        # Generate output path if not provided
        if output_path is None:
            timestamp = int(time.time() * 1000)
            output_path = self.cache_dir / f"tts_{timestamp}.{self.audio_format}"

        # Try providers in priority order
        providers_to_try = self._get_providers_to_try(provider)

        for provider_name in providers_to_try:
            try:
                logger.info(f"Trying TTS with {provider_name}...")

                if provider_name == "elevenlabs":
                    audio_path = await self._generate_elevenlabs(text, voice, str(output_path), **kwargs)
                elif provider_name == "edge_tts":
                    audio_path = await self._generate_edge_tts(text, voice, str(output_path), **kwargs)
                elif provider_name == "espnet":
                    audio_path = await self._generate_espnet(text, voice, str(output_path), **kwargs)
                else:
                    continue

                if audio_path:
                    # Post-process audio
                    processed_path = self._post_process_audio(audio_path)

                    # Cache the result
                    if self.enable_cache:
                        self._cache_audio(text, voice, provider_name, processed_path)

                    generation_time = time.time() - start_time
                    logger.info(f"Audio generated in {generation_time:.2f}s using {provider_name}")
                    return processed_path

            except Exception as e:
                logger.warning(f"TTS failed with {provider_name}: {str(e)}")
                continue

        logger.error("All TTS providers failed")
        return None

    def _get_providers_to_try(self, preferred_provider: str = None) -> List[str]:
        """Get list of providers to try in order"""
        if preferred_provider and preferred_provider in self.providers:
            if self.providers[preferred_provider]["enabled"]:
                return [preferred_provider]
            else:
                logger.warning(f"Preferred provider {preferred_provider} not available")

        # Sort by priority
        available_providers = [
            name for name, config in self.providers.items()
            if config["enabled"]
        ]

        return sorted(available_providers, key=lambda x: self.providers[x]["priority"])

    def _check_cache(self, text: str, voice: str, provider: str) -> Optional[str]:
        """Check if audio is already cached"""
        cache_key = self._get_cache_key(text, voice, provider)
        cache_file = self.cache_dir / f"{cache_key}.{self.audio_format}"

        if cache_file.exists():
            return str(cache_file)
        return None

    def _get_cache_key(self, text: str, voice: str, provider: str) -> str:
        """Generate cache key for text/voice/provider combination"""
        import hashlib

        cache_string = f"{text}_{voice}_{provider}"
        return hashlib.md5(cache_string.encode()).hexdigest()

    def _cache_audio(self, text: str, voice: str, provider: str, audio_path: str):
        """Cache generated audio"""
        try:
            cache_key = self._get_cache_key(text, voice, provider)
            cache_file = self.cache_dir / f"{cache_key}.{self.audio_format}"

            # Copy to cache
            import shutil
            shutil.copy2(audio_path, cache_file)

            # Clean cache if too large
            self._clean_cache()

        except Exception as e:
            logger.warning(f"Failed to cache audio: {str(e)}")

    def _clean_cache(self):
        """Clean cache if it exceeds size limit"""
        try:
            total_size = sum(f.stat().st_size for f in self.cache_dir.glob("*") if f.is_file())
            max_size = self.max_cache_size_mb * 1024 * 1024

            if total_size > max_size:
                # Remove oldest files
                files = [(f, f.stat().st_mtime) for f in self.cache_dir.glob("*") if f.is_file()]
                files.sort(key=lambda x: x[1])  # Sort by modification time

                for file_path, _ in files:
                    file_path.unlink()
                    total_size -= file_path.stat().st_size
                    if total_size <= max_size * 0.8:  # Clean to 80% of limit
                        break

                logger.info("Cache cleaned")

        except Exception as e:
            logger.warning(f"Failed to clean cache: {str(e)}")

    def _post_process_audio(self, audio_path: str) -> str:
        """Post-process audio to ensure consistent format"""
        try:
            # Load audio
            audio, sr = librosa.load(audio_path, sr=None)

            # Resample if needed
            if sr != self.target_sample_rate:
                audio = librosa.resample(audio, orig_sr=sr, target_sr=self.target_sample_rate)

            # Convert to mono if needed
            if len(audio.shape) > 1:
                audio = librosa.to_mono(audio)

            # Normalize audio
            audio = audio / np.max(np.abs(audio))

            # Save processed audio
            processed_path = audio_path.replace(f".{self.audio_format}", f"_processed.{self.audio_format}")
            sf.write(processed_path, audio, self.target_sample_rate)

            return processed_path

        except Exception as e:
            logger.warning(f"Audio post-processing failed: {str(e)}")
            return audio_path

    async def _generate_elevenlabs(self, text: str, voice: str, output_path: str, **kwargs) -> Optional[str]:
        """Generate audio using ElevenLabs"""
        try:
            from elevenlabs import generate

            # Use default voice if not specified
            if not voice:
                voice = "Rachel"

            # Generate audio
            audio_data = generate(
                text=text,
                voice=voice,
                model=kwargs.get("model", "eleven_monolingual_v1"),
                stability=kwargs.get("stability", 0.5),
                similarity_boost=kwargs.get("similarity_boost", 0.5)
            )

            # Save audio
            with open(output_path, "wb") as f:
                f.write(audio_data)

            return output_path

        except Exception as e:
            logger.error(f"ElevenLabs TTS failed: {str(e)}")
            return None

    async def _generate_edge_tts(self, text: str, voice: str, output_path: str, **kwargs) -> Optional[str]:
        """Generate audio using Edge TTS"""
        try:
            import edge_tts

            # Use default voice if not specified
            if not voice:
                voice = "en-US-AriaNeural"

            # Generate audio
            communicate = edge_tts.Communicate(text, voice)
            await communicate.save(output_path)

            return output_path

        except Exception as e:
            logger.error(f"Edge TTS failed: {str(e)}")
            return None

    async def _generate_espnet(self, text: str, voice: str, output_path: str, **kwargs) -> Optional[str]:
        """Generate audio using ESPnet"""
        try:
            # This would be implemented with actual ESPnet TTS
            # For now, create a dummy implementation
            logger.warning("ESPnet TTS not fully implemented")

            # Create a simple sine wave as placeholder
            duration = len(text) * 0.1  # Rough estimate
            sample_rate = self.target_sample_rate
            t = np.linspace(0, duration, int(sample_rate * duration))
            frequency = 440  # A4 note
            audio = 0.3 * np.sin(2 * np.pi * frequency * t)

            # Save audio
            sf.write(output_path, audio, sample_rate)

            return output_path

        except Exception as e:
            logger.error(f"ESPnet TTS failed: {str(e)}")
            return None

    def generate_audio_sync(self, text: str, voice: str = None, provider: str = None,
                           output_path: str = None, **kwargs) -> Optional[str]:
        """
        Synchronous wrapper for generate_audio

        Args:
            text: Text to convert to speech
            voice: Voice to use
            provider: Specific provider to use
            output_path: Output file path
            **kwargs: Additional parameters

        Returns:
            str: Path to generated audio file or None if failed
        """
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        return loop.run_until_complete(
            self.generate_audio(text, voice, provider, output_path, **kwargs)
        )

    def get_stats(self) -> Dict[str, Any]:
        """Get TTS pipeline statistics"""
        cache_files = list(self.cache_dir.glob("*"))
        cache_size = sum(f.stat().st_size for f in cache_files if f.is_file())

        return {
            "providers": self.providers,
            "cache_enabled": self.enable_cache,
            "cache_files": len(cache_files),
            "cache_size_mb": cache_size / (1024 * 1024),
            "target_sample_rate": self.target_sample_rate,
            "target_channels": self.target_channels
        }

    def clear_cache(self):
        """Clear all cached audio files"""
        try:
            for file in self.cache_dir.glob("*"):
                if file.is_file():
                    file.unlink()
            logger.info("Cache cleared")
        except Exception as e:
            logger.error(f"Failed to clear cache: {str(e)}")


# Convenience function for backward compatibility
def generate_audio(text: str, output_path: str = "output/output.wav", voice: str = "Rachel") -> str:
    """
    Simple function for generating audio (backward compatibility)

    Args:
        text: Text to convert to speech
        output_path: Output file path
        voice: Voice to use

    Returns:
        str: Path to generated audio file
    """
    tts = TTSPipeline()
    result = tts.generate_audio_sync(text, voice=voice, output_path=output_path)
    return result or output_path