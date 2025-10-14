# Description: 控制usb相机的基本操作
import cv2


def open_camera(camera_index=0):
    """
    Opens a USB camera and sets the resolution.

    Parameters:
        camera_index (int): Index of the camera to open.

    Returns:
        cap: Opened video capture object.
    """
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise ValueError(f"Unable to open camera with index {camera_index}")

    return cap


def close_camera(cap):
    """
    Releases the camera resource.

    Parameters:
        cap: Video capture object to release.
    """
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    cap = open_camera(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        cv2.imshow("USB Camera", frame)

        # Exit on 'q' key
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    close_camera(cap)
