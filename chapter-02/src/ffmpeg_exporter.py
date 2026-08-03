"""Stream-level video operations via the FFmpeg command line.

This module accompanies Chapter 2 of "Video Generation with AI"
(O'Reilly). It wraps FFmpeg for the dataset hygiene steps that keep
large video collections compact and consistent: compression, trimming,
and aspect-ratio-preserving resizing.

Requires the ffmpeg binary to be installed and on your PATH
(for example, `apt-get install ffmpeg` or `brew install ffmpeg`).
"""

import os
import subprocess


def compress_video(input_path, output_path, codec="libx264", crf=23):
    """Compress a video using FFmpeg and compare file sizes.

    Returns a dict with original size, compressed size (both in MB),
    and the compression ratio.
    """
    cmd = [
        "ffmpeg", "-y",
        "-i", str(input_path),
        "-c:v", codec,
        "-crf", str(crf),
        str(output_path),
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    # Compare file sizes
    original_size = os.path.getsize(input_path) / (1024 ** 2)
    compressed_size = os.path.getsize(output_path) / (1024 ** 2)
    return {
        "original_size_mb": original_size,
        "compressed_size_mb": compressed_size,
        "compression_ratio": original_size / compressed_size,
    }


class FFmpegExporter:
    """Export videos using FFmpeg with various settings."""

    @staticmethod
    def _run(cmd, out_path):
        try:
            subprocess.run(
                cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
            )
            print(f"Saved video to {out_path}")
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            print("Note: FFmpeg not available or the command failed")
            return False

    @staticmethod
    def trim(video_path, start_time, duration, out_path):
        """Extract a segment from a video, keeping the audio track."""
        cmd = [
            "ffmpeg", "-y",
            "-ss", str(start_time),
            "-i", str(video_path),
            "-t", str(duration),
            "-c:v", "libx264",
            "-c:a", "copy",
            str(out_path),
        ]
        return FFmpegExporter._run(cmd, out_path)

    @staticmethod
    def transcode_resize(video_path, out_path, size=(512, 512), crf=23,
                         preset="medium"):
        """Resize and transcode a video, preserving aspect ratio."""
        w, h = size
        vf = f"scale={w}:{h}:force_original_aspect_ratio=decrease"
        cmd = [
            "ffmpeg", "-y",
            "-i", str(video_path),
            "-vf", vf,
            "-c:v", "libx264", "-crf", str(crf), "-preset", preset,
            "-c:a", "copy",
            str(out_path),
        ]
        return FFmpegExporter._run(cmd, out_path)
