#!/usr/bin/env python3
"""YouTube Japanese dubbing pipeline.

Usage:
    python dubbing.py <input> [options]

Examples:
    python dubbing.py "https://youtu.be/VM4eaf3sksE"
    python dubbing.py "https://youtu.be/VM4eaf3sksE" --audio lower --subtitle burn
    python dubbing.py "./video.mp4" --audio mute
    python dubbing.py "https://youtu.be/xxxx" --keep-tmp
"""

import argparse
import logging
import os
import shutil
import sys

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="YouTube Japanese dubbing pipeline",
    )
    parser.add_argument(
        "input",
        help="YouTube URL or local video file path",
    )
    parser.add_argument(
        "--audio",
        choices=["mute", "lower"],
        default="mute",
        help="Original audio handling: mute (remove) or lower (-30dB). Default: mute",
    )
    parser.add_argument(
        "--subtitle",
        choices=["burn", "soft"],
        default="soft",
        help="Subtitle mode: burn (hardcode) or soft (embedded track). Default: soft",
    )
    parser.add_argument(
        "--whisper-model",
        default="large-v3",
        help="Whisper model size. Default: large-v3",
    )
    parser.add_argument(
        "--tts-model",
        default="Qwen/Qwen3-TTS-12Hz-1.7B-Base",
        help="Qwen3-TTS model name. Default: Qwen/Qwen3-TTS-12Hz-1.7B-Base",
    )
    parser.add_argument(
        "--output-dir",
        default="./output",
        help="Output directory. Default: ./output",
    )
    parser.add_argument(
        "--keep-tmp",
        action="store_true",
        help="Keep intermediate files (WAV segments, etc.)",
    )
    return parser.parse_args()


def check_dependencies() -> None:
    """Check that required packages are installed."""
    missing = []
    for module, package in [
        ("whisper", "openai-whisper"),
        ("anthropic", "anthropic"),
        ("torch", "torch"),
        ("soundfile", "soundfile"),
        ("transformers", "transformers"),
    ]:
        try:
            __import__(module)
        except ImportError:
            missing.append(package)
    if missing:
        print(
            f"Error: Missing required packages: {', '.join(missing)}\n"
            f"Install with: pip install {' '.join(missing)}",
            file=sys.stderr,
        )
        sys.exit(1)


def run_pipeline(
    input_path: str,
    output_dir: str = "./output",
    audio_mode: str = "mute",
    subtitle_mode: str = "soft",
    whisper_model: str = "large-v3",
    tts_model: str = "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
    keep_tmp: bool = False,
    on_progress: callable = None,
) -> dict:
    """Run the dubbing pipeline.

    Args:
        input_path: YouTube URL or local video file path.
        output_dir: Output directory.
        audio_mode: "mute" or "lower".
        subtitle_mode: "burn" or "soft".
        whisper_model: Whisper model name.
        tts_model: Qwen3-TTS model name.
        keep_tmp: Whether to keep intermediate files.
        on_progress: Callback(step: int, total: int, message: str).

    Returns:
        Dict with "video", "srt" output paths.
    """
    from pipeline.downloader import download_video, extract_audio, is_url
    from pipeline.transcriber import transcribe
    from pipeline.translator import translate_segments
    from pipeline.tts import generate_dubbed_audio
    from pipeline.composer import compose_video, generate_srt

    logger = logging.getLogger("dubbing")

    def progress(step, total, msg):
        logger.info("=== Step %d/%d: %s ===", step, total, msg)
        if on_progress:
            on_progress(step, total, msg)

    os.makedirs(output_dir, exist_ok=True)
    tmp_dir = os.path.join(output_dir, "tmp")
    os.makedirs(tmp_dir, exist_ok=True)

    try:
        # Step 1: Download / extract audio
        progress(1, 6, "Input processing")
        if is_url(input_path):
            video_path, audio_path, title = download_video(input_path, tmp_dir)
        else:
            if not os.path.isfile(input_path):
                raise FileNotFoundError(f"File not found: {input_path}")
            video_path, audio_path, title = extract_audio(input_path, tmp_dir)

        # Step 2: Transcribe with Whisper
        progress(2, 6, "Transcription (Whisper)")
        segments = transcribe(audio_path, model_name=whisper_model)

        # Step 3: Translate with Claude API
        progress(3, 6, "Translation (Claude API)")
        translated_segments = translate_segments(segments)

        # Step 4: Generate Japanese TTS audio
        progress(4, 6, "TTS (Qwen3-TTS)")
        dubbed_audio_path = generate_dubbed_audio(
            translated_segments, tmp_dir, model_name=tts_model,
        )

        # Step 5: Generate SRT
        progress(5, 6, "Generating subtitles")
        srt_path = os.path.join(output_dir, f"output_{title}.srt")
        generate_srt(translated_segments, srt_path)

        # Step 6: Compose final video
        progress(6, 6, "Composing video (ffmpeg)")
        output_path = os.path.join(output_dir, f"output_{title}.mp4")
        compose_video(
            video_path=video_path,
            dubbed_audio_path=dubbed_audio_path,
            srt_path=srt_path,
            output_path=output_path,
            audio_mode=audio_mode,
            subtitle_mode=subtitle_mode,
        )

        logger.info("Done! Video: %s, SRT: %s", output_path, srt_path)
        return {"video": output_path, "srt": srt_path}

    finally:
        if not keep_tmp and os.path.exists(tmp_dir):
            logger.info("Cleaning up temporary files...")
            shutil.rmtree(tmp_dir)


def main() -> None:
    args = parse_args()

    check_dependencies()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    run_pipeline(
        input_path=args.input,
        output_dir=args.output_dir,
        audio_mode=args.audio,
        subtitle_mode=args.subtitle,
        whisper_model=args.whisper_model,
        tts_model=args.tts_model,
        keep_tmp=args.keep_tmp,
    )


if __name__ == "__main__":
    main()
