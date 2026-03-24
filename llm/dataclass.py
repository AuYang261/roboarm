from dataclasses import dataclass
from typing import Optional
from config_getter import get_config_value


@dataclass
class DetectedFromLLM:
    id: int
    class_name: str
    box_center_x: float
    box_center_y: float
    box_width: float
    box_height: float
    thinking_process: str | None = None
    failed: bool | None = None

    def is_valid(self) -> bool:
        return (
            not self.failed
            and 0.0 <= self.box_center_x <= 1.0
            and 0.0 <= self.box_center_y <= 1.0
            and 0.0 <= self.box_width <= 1.0
            and 0.0 <= self.box_height <= 1.0
        )

    def to_detected_box(
        self,
        img_w: int,
        img_h: int,
        confidence: Optional[float] = None,
        box_rotation_deg: float = 0.0,
    ) -> "DetectedBox":
        if not self.is_valid():
            raise ValueError("Invalid box parameters, cannot convert to DetectedBox.")
        cx = round(self.box_center_x * img_w)
        cy = round(self.box_center_y * img_h)
        if get_config_value("RotationCam2Arm"):
            cx = img_w - cx
            cy = img_h - cy
        w = round(self.box_width * img_w)
        h = round(self.box_height * img_h)

        return DetectedBox(
            class_name=self.class_name,
            box_center_x=cx,
            box_center_y=cy,
            box_width=w,
            box_height=h,
            box_rotation_deg=box_rotation_deg,
            confidence=confidence,
        )


@dataclass
class DetectedBox:
    class_name: str
    box_center_x: int | float
    box_center_y: int | float
    box_width: int | float
    box_height: int | float
    # Optional rotation angle of the bounding box in degrees
    box_rotation_deg: float = 0
    confidence: Optional[float] = None

    def __str__(self) -> str:
        box_center_x_str = (
            f"{self.box_center_x:.2f}"
            if isinstance(self.box_center_x, float)
            else str(self.box_center_x)
        )
        box_center_y_str = (
            f"{self.box_center_y:.2f}"
            if isinstance(self.box_center_y, float)
            else str(self.box_center_y)
        )
        box_width_str = (
            f"{self.box_width:.2f}"
            if isinstance(self.box_width, float)
            else str(self.box_width)
        )
        box_height_str = (
            f"{self.box_height:.2f}"
            if isinstance(self.box_height, float)
            else str(self.box_height)
        )
        box_rotation_deg_str = (
            f"{self.box_rotation_deg:.2f}"
            if isinstance(self.box_rotation_deg, float)
            else str(self.box_rotation_deg)
        )
        conf_str = (
            f", confidence={self.confidence:.2f}" if self.confidence is not None else ""
        )
        return f"DetectedBox(class_name={self.class_name}, box_center=({box_center_x_str}, {box_center_y_str}), box_size=({box_width_str}, {box_height_str}), box_rotation={box_rotation_deg_str} deg{conf_str})"

    def __repr__(self) -> str:
        return self.__str__()
