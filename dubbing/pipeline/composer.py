"""ffmpeg wrapper for composing the final dubbed video."""

import logging
import os
import subprocess

logger = logging.getLogger(__name__)


def generate_srt(segments: list[dict], output_path: str) -> str:
    """Generate an SRT subtitle file from translated segments.

    Args:
        segments: Translated segments [{"start", "end", "text"}, ...].
        output_path: Path to write the SRT file.

    Returns:
        Path to the SRT file.
    """
    with open(output_path, "w", encoding="utf-8") as f:
        for i, seg in enumerate(segments, 1):
            start = _format_srt_time(seg["start"])
            end = _format_srt_time(seg["end"])
            f.write(f"{i}\n{start} --> {end}\n{seg['text']}\n\n")

    logger.info("SRT file saved: %s", output_path)
    return output_path


def _format_srt_time(seconds: float) -> str:
    """Format seconds as SRT timestamp (HH:MM:SS,mmm)."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int((seconds % 1) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def compose_video(
    video_path: str,
    dubbed_audio_path: str,
    srt_path: str,
    output_path: str,
    audio_mode: str = "mute",
    subtitle_mode: str = "soft",
) -> str:
    """Compose final video with dubbed audio and subtitles.

    Args:
        video_path: Path to original video file.
        dubbed_audio_path: Path to dubbed audio WAV.
        srt_path: Path to SRT subtitle file.
        output_path: Path for the output MP4.
        audio_mode: "mute" to remove original audio, "lower" to reduce to -30dB.
        subtitle_mode: "burn" to hardcode subtitles, "soft" to embed as track.

    Returns:
        Path to the output MP4 file.
    """
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    cmd = ["ffmpeg", "-y", "-i", video_path, "-i", dubbed_audio_path]

    filter_parts = []

    # Handle original audio
    if audio_mode == "mute":
        # Map video from input 0, audio from input 1 (dubbed)
        audio_filter = None
        map_args = ["-map", "0:v:0", "-map", "1:a:0"]
    else:
        # Lower original audio to -30dB and mix with dubbed audio
        filter_parts.append("[0:a]volume=-30dB[orig]")
        filter_parts.append("[orig][1:a]amix=inputs=2:duration=longest[aout]")
        audio_filter = "[aout]"
        map_args = ["-map", "0:v:0", "-map", "[aout]"]

    # Handle subtitle burning
    if subtitle_mode == "burn":
        # Escape special characters in path for ffmpeg subtitles filter
        escaped_srt = srt_path.replace("\\", "\\\\").replace(":", "\\:").replace("'", "\\'")
        filter_parts.append(f"[0:v]subtitles='{escaped_srt}'[vout]")
        # Replace video mapping
        map_args[map_args.index("0:v:0")] = "[vout]"

    # Build ffmpeg command
    if filter_parts:
        filter_complex = ";".join(filter_parts)
        cmd.extend(["-filter_complex", filter_complex])
    cmd.extend(map_args)

    # Add soft subtitles
    if subtitle_mode == "soft":
        cmd.extend(["-i", srt_path])
        cmd.extend(["-map", f"{len(cmd) // 2}:s:0"])  # This won't work correctly
        # Rebuild: simpler approach
        cmd = _build_soft_sub_cmd(video_path, dubbed_audio_path, srt_path,
                                   output_path, audio_mode)
    else:
        cmd.extend(["-c:v", "libx264", "-preset", "medium", "-crf", "23"])
        cmd.extend(["-c:a", "aac", "-b:a", "192k"])
        cmd.append(output_path)

    logger.info("Running ffmpeg compose...")
    logger.debug("Command: %s", " ".join(cmd))
    subprocess.run(cmd, check=True, capture_output=True)

    logger.info("Output video saved: %s", output_path)
    return output_path


def _build_soft_sub_cmd(
    video_path: str,
    dubbed_audio_path: str,
    srt_path: str,
    output_path: str,
    audio_mode: str,
) -> list[str]:
    """Build ffmpeg command for soft subtitle embedding."""
    cmd = ["ffmpeg", "-y", "-i", video_path, "-i", dubbed_audio_path, "-i", srt_path]

    if audio_mode == "mute":
        cmd.extend([
            "-map", "0:v:0",
            "-map", "1:a:0",
            "-map", "2:0",
            "-c:v", "copy",
            "-c:a", "aac", "-b:a", "192k",
            "-c:s", "mov_text",
            "-metadata:s:s:0", "language=jpn",
            output_path,
        ])
    else:
        cmd.extend([
            "-filter_complex", "[0:a]volume=-30dB[orig];[orig][1:a]amix=inputs=2:duration=longest[aout]",
            "-map", "0:v:0",
            "-map", "[aout]",
            "-map", "2:0",
            "-c:v", "copy",
            "-c:a", "aac", "-b:a", "192k",
            "-c:s", "mov_text",
            "-metadata:s:s:0", "language=jpn",
            output_path,
        ])

    return cmd
