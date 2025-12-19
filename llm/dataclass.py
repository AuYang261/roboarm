from dataclasses import dataclass
from typing import Optional


@dataclass
class DetectedFromLLM:
    thinking_process: str | None
    failed: bool | None
    id: int
    class_name: str
    box_center_x: float
    box_center_y: float
    box_width: float
    box_height: float

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
    box_center_x: int
    box_center_y: int
    box_width: int
    box_height: int
    # Optional rotation angle of the bounding box in degrees
    box_rotation_deg: float = 0
    confidence: Optional[float] = None
