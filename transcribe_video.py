import argparse
import json
import os
import shutil
from datetime import timedelta
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import torch
import whisper


_FFMPEG_PATH: Path | None = None


def ensure_ffmpeg_available() -> Path:
    """Ensure an ffmpeg binary is discoverable by Whisper."""

    global _FFMPEG_PATH
    if _FFMPEG_PATH and _FFMPEG_PATH.exists():
        return _FFMPEG_PATH

    existing = shutil.which("ffmpeg")
    if existing:
        _FFMPEG_PATH = Path(existing)
        return _FFMPEG_PATH

    try:
        import imageio_ffmpeg  # type: ignore
    except Exception as exc:  # pragma: no cover - import error path
        raise RuntimeError(
            "ffmpeg binary not found. Install ffmpeg or add it to PATH, "
            "or pip install imageio-ffmpeg."
        ) from exc

    ffmpeg_path = Path(imageio_ffmpeg.get_ffmpeg_exe()).resolve()
    if ffmpeg_path.suffix.lower() == ".exe" and ffmpeg_path.name.lower() != "ffmpeg.exe":
        canonical = ffmpeg_path.with_name("ffmpeg.exe")
        if not canonical.exists():  # copy once to provide the expected filename
            shutil.copy(ffmpeg_path, canonical)
        ffmpeg_path = canonical

    os.environ["PATH"] = f"{ffmpeg_path.parent}{os.pathsep}{os.environ.get('PATH', '')}"
    _FFMPEG_PATH = ffmpeg_path
    return _FFMPEG_PATH


def transcribe_audio(model_name: str, video_path: Path, language: str | None) -> Dict:
    """Run Whisper transcription for the given video."""
    ensure_ffmpeg_available()
    model = whisper.load_model(model_name)
    use_fp16 = model.device.type != "cpu"
    result = model.transcribe(str(video_path), language=language, fp16=use_fp16, verbose=False)
    return result


def frame_difference(prev_gray: np.ndarray, gray: np.ndarray) -> float:
    """Compute mean absolute difference between two grayscale frames."""
    diff = cv2.absdiff(prev_gray, gray)
    return float(np.mean(diff))


def extract_keyframes(
    video_path: Path,
    output_dir: Path,
    diff_threshold: float,
    min_interval_sec: float,
    resize_width: int | None,
) -> List[Tuple[int, float, Path]]:
    """Extract frames when the visual difference exceeds threshold."""
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    fps = capture.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        raise RuntimeError("Video FPS is invalid (<=0).")

    frame_records: List[Tuple[int, float, Path]] = []
    prev_gray = None
    last_saved_ts = -min_interval_sec
    frame_idx = 0

    while True:
        ret, frame = capture.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if prev_gray is not None:
            diff_score = frame_difference(prev_gray, gray)
            timestamp_sec = frame_idx / fps

            # Store a frame only when the change is large enough and respects the cooldown period.
            if diff_score >= diff_threshold and timestamp_sec - last_saved_ts >= min_interval_sec:
                save_frame = frame
                if resize_width and resize_width > 0:
                    height = int(frame.shape[0] * resize_width / frame.shape[1])
                    save_frame = cv2.resize(frame, (resize_width, height))

                timestamp_str = format_timedelta(timestamp_sec)
                filename = output_dir / f"frame_{frame_idx:07d}_{timestamp_str}.jpg"
                cv2.imwrite(str(filename), save_frame)
                frame_records.append((frame_idx, timestamp_sec, filename))
                last_saved_ts = timestamp_sec
        prev_gray = gray
        frame_idx += 1

    capture.release()
    return frame_records


def format_timedelta(seconds: float) -> str:
    total_seconds_float = max(seconds, 0.0)
    whole_seconds = int(total_seconds_float)
    ms = int(round((total_seconds_float - whole_seconds) * 1000))
    hours, remainder = divmod(whole_seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if ms == 1000:
        ms = 0
        secs += 1
        if secs == 60:
            secs = 0
            minutes += 1
            if minutes == 60:
                minutes = 0
                hours += 1
    return f"{hours:02d}-{minutes:02d}-{secs:02d}-{ms:03d}"


def write_transcript(transcript: Dict, output_path: Path, output_format: str) -> None:
    if output_format == "json":
        output_path.write_text(json.dumps(transcript, ensure_ascii=False, indent=2), encoding="utf-8")
    elif output_format == "srt":
        segments = transcript.get("segments", [])
        lines: List[str] = []
        for idx, segment in enumerate(segments, start=1):
            start = timedelta(seconds=segment["start"])
            end = timedelta(seconds=segment["end"])
            text = segment["text"].strip()
            lines.append(str(idx))
            lines.append(f"{format_srt_timestamp(start)} --> {format_srt_timestamp(end)}")
            lines.append(text)
            lines.append("")
        output_path.write_text("\n".join(lines), encoding="utf-8")
    elif output_format == "txt":
        output_path.write_text(transcript.get("text", ""), encoding="utf-8")
    else:
        raise ValueError(f"Unsupported transcript output format: {output_format}")


def format_srt_timestamp(td: timedelta) -> str:
    seconds_float = max(td.total_seconds(), 0.0)
    total_seconds = int(seconds_float)
    millis = int(round((seconds_float - total_seconds) * 1000))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if millis == 1000:
        millis = 0
        seconds += 1
        if seconds == 60:
            seconds = 0
            minutes += 1
            if minutes == 60:
                minutes = 0
                hours += 1
    return f"{hours:02d}:{minutes:02d}:{seconds:02d},{millis:03d}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Transcribe MP4 video and extract key frames.")
    parser.add_argument("video", type=Path, nargs="?", default=None, help="Path to the MP4 video file")
    parser.add_argument("output", type=Path, nargs="?", default=None, help="Directory to store transcript and frames")
    parser.add_argument("--model", default="base", help="Whisper model size (tiny, base, small, medium, large)")
    parser.add_argument("--language", default=None, help="Language code to bias transcription (e.g. 'he', 'en')")
    parser.add_argument("--diff-threshold", type=float, default=12.0, help="Mean pixel diff threshold to trigger a screenshot")
    parser.add_argument("--min-interval", type=float, default=2.5, help="Minimum seconds between screenshots")
    parser.add_argument("--resize-width", type=int, default=None, help="Resize saved frames to this width (pixels)")
    parser.add_argument(
        "--transcript-format",
        choices=["json", "srt", "txt"],
        default="json",
        help="Output format for transcript result",
    )
    return parser.parse_args()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def main() -> None:
    args = parse_args()
    video_path: Path | None = args.video
    output_dir: Path | None = args.output

    if video_path is None:
        try:
            user_input = input("Enter the path to the video file: ").strip()
        except EOFError as exc:
            raise SystemExit("No video file provided.") from exc
        if not user_input:
            raise SystemExit("No video file provided.")
        video_path = Path(user_input)

    if output_dir is None:
        output_dir = Path("output") / video_path.stem

    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    ensure_dir(output_dir)
    frames_dir = output_dir / "frames"
    ensure_dir(frames_dir)

    print("Starting transcription...")
    transcript = transcribe_audio(args.model, video_path, args.language)

    transcript_path = output_dir / f"transcript.{args.transcript_format}"
    write_transcript(transcript, transcript_path, args.transcript_format)
    print(f"Transcript saved to {transcript_path}")

    print("Extracting key frames...")
    frames = extract_keyframes(
        video_path=video_path,
        output_dir=frames_dir,
        diff_threshold=args.diff_threshold,
        min_interval_sec=args.min_interval,
        resize_width=args.resize_width,
    )

    metadata = {
        "video_path": str(video_path.resolve()),
        "keyframe_count": len(frames),
        "frames": [
            {
                "frame_index": frame_idx,
                "timestamp_sec": timestamp,
                "timestamp_label": format_timedelta(timestamp),
                "image_path": str(path.relative_to(output_dir)),
            }
            for frame_idx, timestamp, path in frames
        ],
    }

    metadata_path = frames_dir / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Saved {len(frames)} key frames and metadata to {metadata_path}")


if __name__ == "__main__":
    main()
