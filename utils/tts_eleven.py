"""
DEPRECATED: Use utils/tts_pipeline.py instead
This file is kept for backward compatibility
"""

import os
import logging
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)

def generate_audio(text, output_path="output/output.wav"):
    """
    DEPRECATED: Use TTSPipeline from utils/tts_pipeline.py instead
    This function is kept for backward compatibility
    """
    logger.warning("Using deprecated TTS function. Please use TTSPipeline instead.")

    try:
        # Import the new TTS pipeline
        from .tts_pipeline import TTSPipeline

        # Create TTS pipeline instance
        tts = TTSPipeline()

        # Generate audio
        result = tts.generate_audio_sync(text, output_path=output_path)

        if result:
            print(f"[TTS] Audio saved to {result}")
            return result
        else:
            print(f"[TTS] Failed to generate audio")
            return None

    except Exception as e:
        logger.error(f"TTS generation failed: {str(e)}")

        # Fallback to direct ElevenLabs if available
        try:
            api_key = os.getenv("ELEVENLABS_API_KEY")
            if not api_key:
                raise ValueError("ELEVENLABS_API_KEY not found in environment variables")

            from elevenlabs import generate, set_api_key
            set_api_key(api_key)

            audio = generate(
                text=text,
                voice="Rachel",
                model="eleven_monolingual_v1"
            )

            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, "wb") as f:
                f.write(audio)

            print(f"[TTS] Audio saved to {output_path} (fallback)")
            return output_path

        except Exception as fallback_error:
            logger.error(f"Fallback TTS also failed: {str(fallback_error)}")
            print(f"[TTS] All TTS methods failed")
            return None
