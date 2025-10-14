# 仅使用arm摄像头时可用（连接orb摄像头会导致序号改变）
# 导入CV2模块
import cv2
import wmi
import ctypes

def extract_vid_pid(hardware_id: str) -> str:
    """ 从硬件ID中提取VID和PID部分
    :param hardware_id: 硬件ID字符串
    :return: VID和PID部分
    """
    vid_pid = ""
    for item in hardware_id.split('&'):
        if 'VID_' in item or 'PID_' in item:
            vid_pid += item + "&"
    return vid_pid.rstrip("&")

def get_camera_list(class_guid = "{ca3e7ab9-b4c3-4ae6-8251-579ef933890f}") -> list:
    # ref https://blog.csdn.net/miaoyulun/article/details/139206013
    c = wmi.WMI()
    devices = []
    for device in c.Win32_PnPEntity():
        if device.ClassGuid == class_guid:
            devices.append(device)

    table = "debug: camera list\n"
    camera_list = []
    for idx, device in enumerate(devices):
        vid_pid = extract_vid_pid(device.HardwareID[0])
        # 去掉"USB\"前缀
        vid_pid = vid_pid.replace("USB\\", "")
        table += f"{device.Name}\n{vid_pid}\n{idx}\n-------------\n"
        camera_list.append({"index": idx, "name": device.Name, "guid": device.ClassGuid, "vid_pid": vid_pid})
        
    print(table)
    return camera_list

def usb_camera_capture(camera_index = 1):
    # 选择摄像头的编号
    cap = cv2.VideoCapture(camera_index)

    if cap.isOpened():
        # 读取摄像头的画面
        ret, frame = cap.read()
        
        # 保存到当前文件夹下
        img_name = str(camera_index) + "_img.jpg"
        cv2.imwrite(img_name, frame)

    # 释放画面
    cap.release()

def usb_camera_show(camera_index = 1):
    # 选择摄像头的编号
    cap = cv2.VideoCapture(camera_index)
    # 添加这句是可以用鼠标拖动弹出的窗体
    cv2.namedWindow('real_img', cv2.WINDOW_NORMAL)
    
    print("Press 'q' to exit")
    
    while(cap.isOpened()):
        # 读取摄像头的画面
        ret, frame = cap.read()
        # 真实图
        cv2.imshow('real_img', frame)
        # 按下'q'就退出
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    # 释放画面
    cap.release()
    cv2.destroyAllWindows()
    
if __name__ == '__main__':
    
    # 电脑摄像头 guid : {ca3e7ab9-b4c3-4ae6-8251-579ef933890f}
    # arm-camera guid : {ca3e7ab9-b4c3-4ae6-8251-579ef933890f}
    # orb depth : {ca3e7ab9-b4c3-4ae6-8251-579ef933890f}
    # orb ir : {ca3e7ab9-b4c3-4ae6-8251-579ef933890f}
    # orb rgb : {ca3e7ab9-b4c3-4ae6-8251-579ef933890f}
    
    '''
    [
        {'index': 0, 'name': 'Orbbec Gemini 215 RGB Camera', 'guid': '{ca3e7ab9-b4c3-4ae6-8251-579ef933890f}', 'vid_pid': 'VID_2BC5&PID_0808'}
        {'index': 1, 'name': 'Orbbec Gemini 215 IR Camera', 'guid': '{ca3e7ab9-b4c3-4ae6-8251-579ef933890f}', 'vid_pid': 'VID_2BC5&PID_0808'}
        {'index': 2, 'name': 'Integrated Camera', 'guid': '{ca3e7ab9-b4c3-4ae6-8251-579ef933890f}', 'vid_pid': 'VID_174F&PID_246F'}
        {'index': 3, 'name': 'Orbbec Gemini 215 Depth Camera', 'guid': '{ca3e7ab9-b4c3-4ae6-8251-579ef933890f}', 'vid_pid': 'VID_2BC5&PID_0808'}
        {'index': 4, 'name': 'USB2.0_CAM1', 'guid': '{ca3e7ab9-b4c3-4ae6-8251-579ef933890f}', 'vid_pid': 'VID_05A3&PID_9230'}
    ]
    '''
    
    camera_list = get_camera_list()
    print(camera_list)
    
    usb_camera_capture(4)
    
    usb_camera_show(4)
    

