"""Qwen3-TTS wrapper for Japanese speech synthesis."""

import logging
import os

import numpy as np
import soundfile as sf
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

logger = logging.getLogger(__name__)

# Sampling rate used by Qwen3-TTS (12Hz codec, decoded to 24kHz audio)
SAMPLE_RATE = 24000


def load_tts_model(model_name: str, device: str = "cuda:0"):
    """Load Qwen3-TTS model and tokenizer.

    Returns:
        Tuple of (model, tokenizer)
    """
    logger.info("Loading TTS model: %s", model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        torch_dtype=torch.float16,
        device_map=device,
    )
    model.eval()
    logger.info("TTS model loaded on %s", device)
    return model, tokenizer


def synthesize_segment(
    model,
    tokenizer,
    text: str,
    target_duration: float,
    device: str = "cuda:0",
) -> np.ndarray:
    """Synthesize a single text segment to audio.

    Args:
        model: Qwen3-TTS model.
        tokenizer: Qwen3-TTS tokenizer.
        text: Japanese text to synthesize.
        target_duration: Target duration in seconds.
        device: CUDA device.

    Returns:
        Audio samples as numpy array (float32, mono).
    """
    # Build the chat-style prompt for Qwen3-TTS
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {
            "role": "user",
            "content": f"Please speak the following text in Japanese: {text}",
        },
    ]

    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=2048,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
        )

    # Decode generated audio tokens
    new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
    audio_text = tokenizer.decode(new_tokens, skip_special_tokens=True)

    # The model outputs codec tokens that need decoding via the model's decode method
    if hasattr(model, "decode_audio"):
        audio = model.decode_audio(new_tokens)
        if isinstance(audio, torch.Tensor):
            audio = audio.cpu().numpy().flatten().astype(np.float32)
    else:
        # Fallback: try to extract audio from the generated output
        audio = _extract_audio_from_tokens(model, new_tokens, device)

    # Adjust duration to match target
    if len(audio) > 0 and target_duration > 0:
        current_duration = len(audio) / SAMPLE_RATE
        if current_duration > 0:
            # Speed up or slow down to fit target duration
            target_samples = int(target_duration * SAMPLE_RATE)
            if abs(current_duration - target_duration) / target_duration > 0.1:
                # Only adjust if difference is more than 10%
                audio = _resample_to_length(audio, target_samples)

    return audio


def _extract_audio_from_tokens(model, tokens, device: str) -> np.ndarray:
    """Fallback method to extract audio from generated tokens."""
    # Try using the model's built-in audio synthesis pipeline
    try:
        if hasattr(model, "synthesize"):
            result = model.synthesize(tokens)
            if isinstance(result, torch.Tensor):
                return result.cpu().numpy().flatten().astype(np.float32)
            return np.array(result, dtype=np.float32).flatten()
    except Exception as e:
        logger.warning("Audio extraction fallback failed: %s", e)

    # Return silence if we can't extract audio
    logger.warning("Could not extract audio from tokens, returning silence")
    return np.zeros(SAMPLE_RATE, dtype=np.float32)  # 1 second of silence


def _resample_to_length(audio: np.ndarray, target_length: int) -> np.ndarray:
    """Resample audio to target length using linear interpolation."""
    if len(audio) == 0 or target_length <= 0:
        return audio
    indices = np.linspace(0, len(audio) - 1, target_length)
    return np.interp(indices, np.arange(len(audio)), audio).astype(np.float32)


def generate_dubbed_audio(
    segments: list[dict],
    tmp_dir: str,
    model_name: str = "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
    device: str = "cuda:0",
) -> str:
    """Generate full dubbed audio track from translated segments.

    Args:
        segments: Translated segments [{"start", "end", "text"}, ...].
        tmp_dir: Directory for temporary WAV files.
        model_name: Qwen3-TTS model name.
        device: CUDA device.

    Returns:
        Path to the combined WAV file.
    """
    os.makedirs(tmp_dir, exist_ok=True)

    model, tokenizer = load_tts_model(model_name, device)

    # Determine total duration from the last segment
    total_duration = max(seg["end"] for seg in segments)
    total_samples = int(total_duration * SAMPLE_RATE)
    combined = np.zeros(total_samples, dtype=np.float32)

    for i, seg in enumerate(segments):
        logger.info(
            "Generating TTS segment %d/%d: %.1fs-%.1fs",
            i + 1, len(segments), seg["start"], seg["end"],
        )

        target_duration = seg["end"] - seg["start"]
        audio = synthesize_segment(
            model, tokenizer, seg["text"], target_duration, device
        )

        # Place audio at the correct position
        start_sample = int(seg["start"] * SAMPLE_RATE)
        end_sample = start_sample + len(audio)

        # Ensure we don't exceed the combined array
        if end_sample > total_samples:
            audio = audio[:total_samples - start_sample]
            end_sample = total_samples

        if len(audio) > 0:
            combined[start_sample:start_sample + len(audio)] = audio

        # Save individual segment WAV for debugging
        seg_path = os.path.join(tmp_dir, f"seg_{i:04d}.wav")
        sf.write(seg_path, audio, SAMPLE_RATE)

    # Save combined audio
    combined_path = os.path.join(tmp_dir, "dubbed_audio.wav")
    sf.write(combined_path, combined, SAMPLE_RATE)
    logger.info("Combined dubbed audio saved: %s", combined_path)

    return combined_path
