import time
import base64
from collections.abc import Sequence

import cv2
import numpy as np
import requests
from utils.config_getter import get_config_value


class SimArmClient:
    def __init__(self, host: str, port: int):
        self.base_url = f"http://{host}:{port}"
        self.timeout_s = get_config_value(
            "arm_sim_timeout_s", 3.0, raise_if_missing=False
        )

    def _request(
        self,
        method: str,
        path: str,
        payload: dict | None = None,
    ) -> dict:
        try:
            resp = requests.request(
                method,
                f"{self.base_url}{path}",
                json=payload,
                timeout=self.timeout_s,
            )
        except requests.ConnectionError as exc:
            raise ConnectionError(f"simulation service is unavailable: {exc}") from exc
        except requests.Timeout as exc:
            raise ConnectionError(
                f"simulation service request timed out: {exc}"
            ) from exc

        if not resp.ok:
            raise ConnectionError(
                f"simulation service request failed: {resp.status_code}; {resp.text}"
            )

        if not resp.text:
            return {}

        result = resp.json()
        if isinstance(result, dict) and result.get("ok") is False:
            raise RuntimeError(result.get("error") or "仿真服务返回失败")
        return result

    def ping(self) -> dict:
        return self._request("GET", "/health")

    def get_joint_names(self) -> list[str]:
        result = self._request("GET", "/state")
        joint_names = result.get("joint_names")
        if not isinstance(joint_names, list) or not all(
            isinstance(name, str) for name in joint_names
        ):
            raise RuntimeError("仿真服务未返回合法的 joint_names")
        return joint_names

    def get_raw_joint_angles(self) -> tuple[list[float], float]:
        result = self._request("GET", "/state")
        joint_angles = result.get("joint_angles_deg")
        gripper_open_0to1 = result.get("gripper_open_0to1")
        if not isinstance(joint_angles, list) or not all(
            isinstance(angle, (int, float)) for angle in joint_angles
        ):
            raise RuntimeError("仿真服务未返回合法的 joint_angles_deg")
        if not isinstance(gripper_open_0to1, (int, float)):
            raise RuntimeError("仿真服务未返回合法的 gripper_open_0to1")
        return [float(angle) for angle in joint_angles], float(gripper_open_0to1)

    def send_joint_targets(
        self,
        joint_names: Sequence[str],
        joint_angles_deg: Sequence[float],
        gripper_open_0to1: float,
    ) -> None:
        self._request(
            "POST",
            "/state",
            {
                "joint_names": list(joint_names),
                "joint_angles_deg": [float(angle) for angle in joint_angles_deg],
                "gripper_open_0to1": float(gripper_open_0to1),
            },
        )

    def set_torque_enabled(self, enabled: bool) -> None:
        self._request("POST", "/torque", {"enabled": bool(enabled)})

    def spawn_object(
        self,
        position: Sequence[float],
        object_type: str = "block",
        rotation_rad: float | None = None,
    ) -> dict:
        payload = {
            "type": object_type,
            "position": [float(value) for value in position],
        }
        if rotation_rad is not None:
            payload["rotation_rad"] = float(rotation_rad)
        while True:
            try:
                ret = self._request("POST", "/object", payload)
                break
            except ConnectionError as e:
                print(e)
                time.sleep(1)
        return ret

    def get_object_pose(self) -> list[float]:
        result = self._request("GET", "/object/pose")
        pose = result.get("position", result.get("object_position"))
        if not isinstance(pose, list) or not all(
            isinstance(value, (int, float)) for value in pose
        ):
            raise RuntimeError("simulation service did not return object position")
        return [float(value) for value in pose]

    def get_camera_rgb(self, camera_name: str = "gripper_cam") -> np.ndarray:
        result = self._request("GET", f"/camera/rgb?camera={camera_name}")
        image = result.get("rgb", result.get("image"))
        if isinstance(image, list):
            return np.asarray(image, dtype=np.uint8)
        if isinstance(image, str):
            raw = base64.b64decode(image)
            decoded = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR)
            if decoded is None:
                raise RuntimeError("simulation service returned an invalid rgb image")
            return cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB)
        raise RuntimeError("simulation service did not return rgb image data")

    def disconnect(self) -> None:
        # self._request("POST", "/shutdown", {})
        pass

    def shutdown(self) -> None:
        self._request("POST", "/shutdown", {})
