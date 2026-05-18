import argparse
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import h5py
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from arm.arm_base import Arm
from utils.config_getter import get_config_value
from sim.sim_client import SimArmClient


@dataclass
class EpisodeBuffer:
    target_joint_angles_deg: list[list[float]] = field(default_factory=list)
    target_gripper_open_0to1: list[float] = field(default_factory=list)
    current_joint_angles_deg: list[list[float]] = field(default_factory=list)
    current_gripper_open_0to1: list[float] = field(default_factory=list)
    camera_rgb: list[np.ndarray] = field(default_factory=list)
    object_position: list[list[float]] = field(default_factory=list)
    timestamp_s: list[float] = field(default_factory=list)


def _float_pair(value: str) -> tuple[float, float]:
    parts = [part.strip() for part in value.split(",")]
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("range must be formatted as min,max")
    low, high = float(parts[0]), float(parts[1])
    if low > high:
        raise argparse.ArgumentTypeError("range min must be <= max")
    return low, high


def _write_episode(
    path: Path,
    buffer: EpisodeBuffer,
    *,
    success: bool,
    arm_success: bool,
    object_spawn_info: dict[str, Any],
    object_start_position: list[float],
    object_final_position: list[float] | None,
    place_position: list[float],
    error: str | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as h5:
        h5.create_dataset(
            "target_joint_angles_deg",
            data=np.asarray(buffer.target_joint_angles_deg, dtype=np.float32),
        )
        h5.create_dataset(
            "target_gripper_open_0to1",
            data=np.asarray(buffer.target_gripper_open_0to1, dtype=np.float32),
        )
        h5.create_dataset(
            "current_joint_angles_deg",
            data=np.asarray(buffer.current_joint_angles_deg, dtype=np.float32),
        )
        h5.create_dataset(
            "current_gripper_open_0to1",
            data=np.asarray(buffer.current_gripper_open_0to1, dtype=np.float32),
        )
        if buffer.camera_rgb:
            h5.create_dataset(
                "camera_rgb",
                data=np.stack(buffer.camera_rgb).astype(np.uint8),
                compression="gzip",
                compression_opts=4,
                chunks=(1, *buffer.camera_rgb[0].shape),
            )
        else:
            h5.create_dataset("camera_rgb", data=np.empty((0,), dtype=np.uint8))
        h5.create_dataset(
            "object_position",
            data=np.asarray(buffer.object_position, dtype=np.float32),
        )
        h5.create_dataset(
            "timestamp_s",
            data=np.asarray(buffer.timestamp_s, dtype=np.float64),
        )
        h5.create_dataset("success", data=np.asarray(success, dtype=np.bool_))
        h5.create_dataset("arm_success", data=np.asarray(arm_success, dtype=np.bool_))
        h5.attrs["success"] = bool(success)
        h5.attrs["arm_success"] = bool(arm_success)
        h5.attrs["object_start_position"] = np.asarray(
            object_start_position, dtype=np.float32
        )
        if object_final_position is not None:
            h5.attrs["object_final_position"] = np.asarray(
                object_final_position, dtype=np.float32
            )
        h5.attrs["place_position"] = np.asarray(place_position, dtype=np.float32)
        h5.attrs["object_spawn_info"] = str(object_spawn_info)
        if error:
            h5.attrs["error"] = error


def _is_place_success(
    object_position: list[float] | None,
    place_position: list[float],
    threshold_m: float,
) -> bool:
    if object_position is None:
        return False
    dim = min(len(object_position), len(place_position), 2)
    if dim == 0:
        return False
    distance = np.linalg.norm(
        np.asarray(object_position[:dim]) - np.asarray(place_position[:dim])
    )
    return bool(distance <= threshold_m)


def collect_episode(
    arm: Arm,
    episode_index: int,
    output_dir: Path,
    rng: np.random.Generator,
    args: argparse.Namespace,
) -> Path:
    sim_client: SimArmClient = getattr(arm, "sim_arm_client")
    if sim_client is None:
        raise RuntimeError(
            "data_produce.py requires arm_type=lerobo and arm_backend=sim"
        )

    object_position = [
        float(rng.uniform(*args.object_x_range)),
        float(rng.uniform(*args.object_y_range)),
        float(args.object_z),
    ]
    object_rotation = float(rng.uniform(-np.pi, np.pi))
    spawn_info = sim_client.spawn_object(
        object_position,
        object_type=args.object_type,
        rotation_rad=object_rotation,
    )

    buffer = EpisodeBuffer()
    start_time = time.monotonic()

    def record_frame(step: dict[str, Any]) -> None:
        current_angles, current_gripper = arm.get_arm_angles()
        if current_angles is None or current_gripper is None:
            raise RuntimeError("failed to read current joint state")
        target_angles = step["target_joint_angles_deg"]
        buffer.target_joint_angles_deg.append([float(v) for v in target_angles])
        buffer.target_gripper_open_0to1.append(float(step["target_gripper_open_0to1"]))
        buffer.current_joint_angles_deg.append([float(v) for v in current_angles])
        buffer.current_gripper_open_0to1.append(float(current_gripper))
        buffer.camera_rgb.append(sim_client.get_camera_rgb())
        buffer.object_position.append(sim_client.get_object_pose())
        buffer.timestamp_s.append(time.monotonic() - start_time)

    place_position = [float(value) for value in args.place_position]
    arm_success = False
    final_position = None
    error = None
    try:
        arm.move_to_home(gripper_open_0to1=1, step_callback=record_frame)
        arm_success = arm.catch_and_place(
            object_position[0],
            object_position[1],
            object_rotation,
            place_pos=place_position,
            height=object_position[2],
            place_rotate_rad=args.place_rotation_rad,
            step_callback=record_frame,
        )
        final_position = sim_client.get_object_pose()
    except Exception as exc:
        error = str(exc)
        try:
            final_position = sim_client.get_object_pose()
        except Exception:
            final_position = None

    success = arm_success and _is_place_success(
        final_position,
        place_position,
        args.success_distance_threshold,
    )
    path = output_dir / f"episode_{episode_index:06d}.hdf5"
    _write_episode(
        path,
        buffer,
        success=success,
        arm_success=arm_success,
        object_spawn_info=spawn_info,
        object_start_position=object_position,
        object_final_position=final_position,
        place_position=place_position,
        error=error,
    )
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Produce simulated grasp episodes.")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--output-dir", type=Path, default=Path("sim_dataset"))
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--object-type", default="block")
    parser.add_argument("--object-x-range", type=_float_pair, default=(-0.08, 0.08))
    parser.add_argument("--object-y-range", type=_float_pair, default=(-0.08, 0.08))
    parser.add_argument(
        "--object-z",
        type=float,
        default=float(get_config_value("default_desktop_height", 0.075)),
    )
    parser.add_argument("--place-position", type=float, nargs="+", default=[0.2, 0.0])
    parser.add_argument("--place-rotation-rad", type=float, default=0.0)
    parser.add_argument(
        "--success-distance-threshold",
        type=float,
        default=float(
            get_config_value("place_distance_threshold", 0.05, raise_if_missing=False)
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    arm = Arm()
    try:
        for episode_index in range(args.episodes):
            path = collect_episode(arm, episode_index, args.output_dir, rng, args)
            print(f"saved {path}")
    finally:
        arm.disconnect_arm()


if __name__ == "__main__":
    main()
