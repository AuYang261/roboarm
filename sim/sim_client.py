import time
from collections.abc import Sequence

import requests
from utils.config_getter import get_config_value


class SimArmClient:
    def __init__(self, host: str, port: int):
        self.base_url = f"http://{host}:{port}"
        self.timeout_s = get_config_value(
            "arm_sim_timeout_s", 3.0, raise_if_missing=False
        )
        self.reach_mse_threshold = float(
            get_config_value(
                "arm_sim_reach_mse_threshold_deg2", 1.0, raise_if_missing=False
            )
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

    def wait_until_reached(self, target_angles_deg: Sequence[float]) -> None:
        """轮询 /state 直到当前关节角与目标关节角的均方差小于阈值，或超时。

        Args:
            target_angles_deg: 期望最终到达的关节角列表，单位为度。

        Raises:
            TimeoutError: 在 `timeout_s` 内未达到阈值。
            RuntimeError: 仿真服务返回的关节数与目标关节数不一致。
        """
        target = [float(angle) for angle in target_angles_deg]
        deadline = time.monotonic() + self.timeout_s
        while True:
            current_angles_deg, _ = self.get_raw_joint_angles()
            if len(current_angles_deg) != len(target):
                raise RuntimeError(
                    "仿真服务返回的关节数与目标关节数不一致: "
                    f"{len(current_angles_deg)} vs {len(target)}"
                )
            squared_errors = [
                (current - desired) ** 2
                for current, desired in zip(current_angles_deg, target, strict=True)
            ]
            mse = sum(squared_errors) / len(squared_errors)
            if mse <= self.reach_mse_threshold:
                return
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    f"等待仿真机械臂到位超时: 当前 MSE={mse:.4f} deg^2, "
                    f"阈值={self.reach_mse_threshold} deg^2, "
                    f"超时={self.timeout_s}s"
                )
            time.sleep(0.1)

    def set_torque_enabled(self, enabled: bool) -> None:
        self._request("POST", "/torque", {"enabled": bool(enabled)})

    def disconnect(self) -> None:
        # self._request("POST", "/shutdown", {})
        pass

    def shutdown(self) -> None:
        self._request("POST", "/shutdown", {})
