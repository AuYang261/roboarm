"""Convert HDF5 episode files to videos for each camera view."""

import argparse
from pathlib import Path
from typing import Any

import cv2
import h5py
import numpy as np


def h5_to_video(
    h5_path: Path,
    output_dir: Path | None = None,
    fps: int = 30,
    camera_views: list[str] | None = None,
) -> dict[str, Path]:
    """Convert camera frames in HDF5 file to video files.

    Args:
        h5_path: Path to the HDF5 episode file
        output_dir: Output directory for videos (default: same as h5 file)
        fps: Frames per second for output videos
        camera_views: List of camera view names to convert (default: all available)

    Returns:
        Dictionary mapping camera view names to output video paths
    """
    if output_dir is None:
        output_dir = h5_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    if camera_views is None:
        camera_views = ["camera_rgb", "camera_top_rgb"]

    output_paths = {}

    with h5py.File(h5_path, "r") as h5:
        episode_name = h5_path.stem

        for camera_name in camera_views:
            if camera_name not in h5:
                print(f"Warning: {camera_name} not found in {h5_path.name}, skipping")
                continue

            frames = h5[camera_name][:]

            if frames.size == 0 or len(frames.shape) != 4:
                print(
                    f"Warning: {camera_name} has invalid shape {frames.shape}, skipping"
                )
                continue

            num_frames, height, width, channels = frames.shape

            if num_frames == 0:
                print(f"Warning: {camera_name} has no frames, skipping")
                continue

            # Create output video path
            video_name = f"{episode_name}_{camera_name}.mp4"
            video_path = output_dir / video_name

            # Initialize video writer
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(
                str(video_path),
                fourcc,
                fps,
                (width, height),
            )

            if not writer.isOpened():
                print(f"Error: Failed to open video writer for {video_path}")
                continue

            # Write frames
            for i in range(num_frames):
                frame = frames[i]
                # Convert RGB to BGR for OpenCV
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                writer.write(frame_bgr)

            writer.release()
            output_paths[camera_name] = video_path
            print(f"Created {video_path} ({num_frames} frames at {fps} fps)")

    return output_paths


def batch_convert(
    input_dir: Path,
    output_dir: Path | None = None,
    fps: int = 30,
    camera_views: list[str] | None = None,
    pattern: str = "*.hdf5",
) -> None:
    """Convert all HDF5 files in a directory to videos.

    Args:
        input_dir: Directory containing HDF5 episode files
        output_dir: Output directory for videos (default: input_dir/videos)
        fps: Frames per second for output videos
        camera_views: List of camera view names to convert (default: all available)
        pattern: Glob pattern for matching HDF5 files
    """
    if output_dir is None:
        output_dir = input_dir / "videos"

    h5_files = sorted(input_dir.glob(pattern))

    if not h5_files:
        print(f"No files matching '{pattern}' found in {input_dir}")
        return

    print(f"Found {len(h5_files)} HDF5 files to convert")

    for h5_path in h5_files:
        print(f"\nProcessing {h5_path.name}...")
        try:
            h5_to_video(h5_path, output_dir, fps, camera_views)
        except Exception as e:
            print(f"Error processing {h5_path.name}: {e}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert HDF5 episode files to videos for each camera view."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path(__file__).parent / "sim_dataset",
        help="Input HDF5 file or directory containing HDF5 files",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory for videos (default: same as input or input/videos)",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=30,
        help="Frames per second for output videos (default: 30)",
    )
    parser.add_argument(
        "--camera-views",
        nargs="+",
        default=None,
        help="Camera view names to convert (default: all available)",
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default="*.hdf5",
        help="Glob pattern for matching HDF5 files when input is a directory (default: *.hdf5)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    input_path = args.input

    if not input_path.exists():
        print(f"Error: {input_path} does not exist")
        return

    if input_path.is_file():
        # Convert single file
        h5_to_video(
            input_path,
            args.output_dir,
            args.fps,
            args.camera_views,
        )
    elif input_path.is_dir():
        # Batch convert directory
        batch_convert(
            input_path,
            args.output_dir,
            args.fps,
            args.camera_views,
            args.pattern,
        )
    else:
        print(f"Error: {input_path} is neither a file nor a directory")


if __name__ == "__main__":
    main()
