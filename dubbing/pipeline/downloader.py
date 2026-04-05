"""yt-dlp wrapper for downloading videos and extracting audio."""

import os
import subprocess
import re
import logging

logger = logging.getLogger(__name__)


def sanitize_filename(name: str) -> str:
    """Remove or replace characters that are unsafe for filenames."""
    name = re.sub(r'[<>:"/\\|?*]', "_", name)
    name = name.strip(". ")
    return name[:200] if name else "video"


def is_url(input_path: str) -> bool:
    """Check if input is a URL."""
    return input_path.startswith("http://") or input_path.startswith("https://")


def download_video(url: str, tmp_dir: str) -> tuple[str, str, str]:
    """Download video from YouTube URL.

    Returns:
        Tuple of (video_path, audio_path, title)
    """
    os.makedirs(tmp_dir, exist_ok=True)

    # Get video title first
    result = subprocess.run(
        ["yt-dlp", "--get-title", "--no-warnings", url],
        capture_output=True, text=True, check=True,
    )
    title = sanitize_filename(result.stdout.strip())
    logger.info("Video title: %s", title)

    video_path = os.path.join(tmp_dir, f"{title}.mp4")
    audio_path = os.path.join(tmp_dir, f"{title}.mp3")

    # Download video as MP4
    logger.info("Downloading video...")
    subprocess.run(
        [
            "yt-dlp",
            "-f", "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best",
            "--merge-output-format", "mp4",
            "-o", video_path,
            "--no-warnings",
            url,
        ],
        check=True,
    )

    # Extract audio as MP3
    logger.info("Extracting audio...")
    subprocess.run(
        [
            "ffmpeg", "-y", "-i", video_path,
            "-vn", "-acodec", "libmp3lame", "-q:a", "2",
            audio_path,
        ],
        check=True, capture_output=True,
    )

    return video_path, audio_path, title


def extract_audio(video_path: str, tmp_dir: str) -> tuple[str, str, str]:
    """Extract audio from a local video file.

    Returns:
        Tuple of (video_path, audio_path, title)
    """
    os.makedirs(tmp_dir, exist_ok=True)

    title = sanitize_filename(os.path.splitext(os.path.basename(video_path))[0])
    audio_path = os.path.join(tmp_dir, f"{title}.mp3")

    logger.info("Extracting audio from local file...")
    subprocess.run(
        [
            "ffmpeg", "-y", "-i", video_path,
            "-vn", "-acodec", "libmp3lame", "-q:a", "2",
            audio_path,
        ],
        check=True, capture_output=True,
    )

    return os.path.abspath(video_path), audio_path, title
