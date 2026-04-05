"""Whisper wrapper for speech-to-text transcription."""

import logging

import whisper

logger = logging.getLogger(__name__)


def transcribe(audio_path: str, model_name: str = "large-v3") -> list[dict]:
    """Transcribe audio file using Whisper.

    Args:
        audio_path: Path to the MP3 audio file.
        model_name: Whisper model size to use.

    Returns:
        List of segments: [{"start": float, "end": float, "text": str}, ...]
    """
    logger.info("Loading Whisper model: %s", model_name)
    model = whisper.load_model(model_name)

    logger.info("Transcribing audio...")
    result = model.transcribe(audio_path, verbose=False)

    segments = []
    for seg in result["segments"]:
        segments.append({
            "start": seg["start"],
            "end": seg["end"],
            "text": seg["text"].strip(),
        })

    logger.info("Transcribed %d segments", len(segments))
    return segments
