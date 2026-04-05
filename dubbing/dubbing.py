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


def main() -> None:
    args = parse_args()

    # Check dependencies before importing pipeline modules
    check_dependencies()

    from pipeline.downloader import download_video, extract_audio, is_url
    from pipeline.transcriber import transcribe
    from pipeline.translator import translate_segments
    from pipeline.tts import generate_dubbed_audio
    from pipeline.composer import compose_video, generate_srt

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    logger = logging.getLogger("dubbing")

    # Setup directories
    os.makedirs(args.output_dir, exist_ok=True)
    tmp_dir = os.path.join(args.output_dir, "tmp")
    os.makedirs(tmp_dir, exist_ok=True)

    try:
        # Step 1: Download / extract audio
        logger.info("=== Step 1: Input processing ===")
        if is_url(args.input):
            video_path, audio_path, title = download_video(args.input, tmp_dir)
        else:
            if not os.path.isfile(args.input):
                logger.error("File not found: %s", args.input)
                sys.exit(1)
            video_path, audio_path, title = extract_audio(args.input, tmp_dir)

        # Step 2: Transcribe with Whisper
        logger.info("=== Step 2: Transcription (Whisper) ===")
        segments = transcribe(audio_path, model_name=args.whisper_model)

        # Step 3: Translate with Claude API
        logger.info("=== Step 3: Translation (Claude API) ===")
        translated_segments = translate_segments(segments)

        # Step 4: Generate Japanese TTS audio
        logger.info("=== Step 4: TTS (Qwen3-TTS) ===")
        dubbed_audio_path = generate_dubbed_audio(
            translated_segments, tmp_dir,
            model_name=args.tts_model,
        )

        # Step 5: Generate SRT
        logger.info("=== Step 5: Generating subtitles ===")
        srt_path = os.path.join(args.output_dir, f"output_{title}.srt")
        generate_srt(translated_segments, srt_path)

        # Step 6: Compose final video
        logger.info("=== Step 6: Composing video (ffmpeg) ===")
        output_path = os.path.join(args.output_dir, f"output_{title}.mp4")
        compose_video(
            video_path=video_path,
            dubbed_audio_path=dubbed_audio_path,
            srt_path=srt_path,
            output_path=output_path,
            audio_mode=args.audio,
            subtitle_mode=args.subtitle,
        )

        logger.info("=== Done! ===")
        logger.info("Output video: %s", output_path)
        logger.info("Subtitles: %s", srt_path)

    finally:
        # Cleanup tmp directory unless --keep-tmp is set
        if not args.keep_tmp and os.path.exists(tmp_dir):
            logger.info("Cleaning up temporary files...")
            shutil.rmtree(tmp_dir)


if __name__ == "__main__":
    main()
