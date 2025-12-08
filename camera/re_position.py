
import cv2

import numpy as np

# 使用机械臂摄像头对机械臂进行精确定位校准操作
def main():

    camera_index = 1
    cap = cv2.VideoCapture(camera_index)
    
    while(cap.isOpened()):
        # 读取摄像头的画面
        ret, frame = cap.read()
        if frame is None:
            continue
        org_frame = frame.copy()
        # 图片增加对比度
        # frame = cv2.convertScaleAbs(frame, alpha=1.5, beta=0)
        
        # 图像添加mask
        """
Point coordinates: (53, 251)
Point coordinates: (1, 271)
Point coordinates: (4, 476)
Point coordinates: (619, 474)
Point coordinates: (465, 253)
Point coordinates: (407, 250)
Point coordinates: (361, 217)
Point coordinates: (339, 219)
Point coordinates: (330, 298)
Point coordinates: (368, 358)
Point coordinates: (369, 406)
Point coordinates: (254, 404)
Point coordinates: (222, 374)
Point coordinates: (96, 293)
Point coordinates: (76, 245)
Point coordinates: (50, 251)

        """
        # 定义多边形顶点
        pts = np.array([[53, 251], [1, 271], [4, 476], [619, 474], [465, 253],
                        [407, 250], [361, 217], [339, 219], [330, 298], [368, 358],
                        [369, 406], [254, 404], [222, 374], [96, 293], [76, 245]], np.int32)
        cv2.polylines(frame, [pts], isClosed=True, color=(0, 0, 255), thickness=2)

        

        # 定义第二个多边形
        '''
        Point coordinates: (76, 121)
        Point coordinates: (80, 238)
        Point coordinates: (102, 288)
        Point coordinates: (254, 390)
        Point coordinates: (357, 394)
        Point coordinates: (326, 310)
        Point coordinates: (333, 215)
        Point coordinates: (337, 125)
        Point coordinates: (81, 128)
        '''
        pts2 = np.array([[76, 121], [80, 238], [102, 288], [254, 390],
                         [357, 394], [326, 310], [333, 215], [337, 125]], np.int32)
        # frame 中只保留第二个多边形中的区域
        mask = np.zeros_like(frame)
        cv2.fillPoly(mask, [pts2], (255, 255, 255))
        frame = cv2.bitwise_and(frame, mask)
        
        # 展示添加mask后的图像
        cv2.imshow('Masked Frame', frame)
        
        # 转为灰度图
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # 高斯模糊
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        # 边缘检测
        edged = cv2.Canny(blurred, 50, 150)
        cv2.imshow('Edged Frame', edged)
        

        # 原始图像上绘制多边形1
        cv2.polylines(org_frame, [pts], isClosed=True, color=(0, 0, 255), thickness=2)
        # 原始图像上绘制多边形2
        cv2.polylines(org_frame, [pts2], isClosed=True, color=(0, 255, 0), thickness=2)
        # 原始图像上绘制轮廓
        
        
        cv2.imshow('Arm Camera Contours', org_frame)
        
        # 获取图像上一点的坐标
        def get_point(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                print(f"Point coordinates: ({x}, {y})")
        cv2.setMouseCallback('Arm Camera Contours', get_point)
        
        
        
        
    
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

if __name__ == "__main__":
    main()
    