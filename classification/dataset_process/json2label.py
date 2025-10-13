# Description: 将LabelMe生成的json标签文件转换为yolo obb要求的txt文件并预览
import os
import json
import shutil
import numpy as np
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm
import yaml

class_names = list(
    yaml.safe_load(
        open(
            os.path.join(
                os.path.dirname(os.path.dirname(__file__)), "object_detect", "data.yaml"
            ),
            "r",
        )
    )["names"].values()
)


def get_file_list(path, file_list):
    """
    递归地获取指定路径下所有.json文件的路径列表。

    本函数会遍历指定的目录，寻找所有的.json文件，并将它们的路径添加到一个列表中。
    如果找到了对应的图片文件（通过替换.json为.png得到图片文件名），则只添加.json文件的路径到列表中；
    否则，会打印出那些没有对应图片文件的.json文件的路径。

    参数:
    path (str): 要遍历的目录路径。
    file_list (list): 用于存储找到的.json文件路径的列表。

    返回:
    list: 包含所有找到的.json文件路径的列表。
    """
    # 获取当前路径下的所有文件和文件夹
    file_paths = os.listdir(path)

    # 遍历当前路径下的所有文件和文件夹
    for file in file_paths:
        # 构成文件或文件夹的完整路径
        file_path = os.path.join(path, file)

        # 如果是文件夹，则递归调用本函数
        if os.path.isdir(file_path):
            get_file_list(file_path, file_list)
        else:
            # 如果是.json文件，则进行进一步检查
            if file_path.endswith(".json"):
                # 通过替换.json为.png得到对应图片文件的名称
                img_name = file_path.replace(".json", ".png")

                # 如果对应的图片文件存在于文件列表中，则添加.json文件的路径到列表中
                if os.path.basename(img_name) in file_paths:
                    file_list.append(file_path)
                else:
                    # 如果没有找到对应的图片文件，则打印.json文件的路径
                    print(file_path)

    # 返回包含所有找到的.json文件路径的列表
    return file_list


def labelme_to_obb(
    img_path: str, label_path: str, save_dir: str, if_test: bool = False
):
    """
    将LabelMe标注工具生成的实例分割标签文件转换为yolo obb检测标签格式，并保存为TXT文件。

    参数:
    img_path: 图像文件路径。
    label_path: LabelMe生成的JSON标签文件路径。
    save_dir: 转换后的TXT标签文件保存目录。
    if_test: 是否生成测试图像，默认为False。

    返回:
    成功转换并保存标签文件时返回True，否则返回False。
    """
    # 读取JSON标签文件
    with open(label_path, "r") as f:
        label_data = json.load(f)
        shapes = label_data["shapes"]
        img_w = label_data["imageWidth"]
        img_h = label_data["imageHeight"]
        img_w_h = np.array([[img_w, img_h]])

    label_list = []
    for shape in shapes:
        points = np.array(shape["points"], dtype=np.int32)
        # 检查标签是否在定义的类别名称中
        if shape["label"] not in class_names:
            print("label %s not in classes" % shape["label"], label_path)
            label_list = None
            break
        # 根据标签分配类别ID
        label = class_names.index(shape["label"])

        # 计算最小面积外接矩形
        rect = cv2.minAreaRect(points)
        box = cv2.boxPoints(rect)
        obb = box / img_w_h
        shape["points"] = obb.tolist()
        rex = np.round(obb, 5)
        # 将二维数组转为字符串
        rex = rex.reshape(-1)
        points_nor_str = " ".join(map(str, rex))
        label_str = str(label) + " " + points_nor_str + "\n"
        label_list.append(label_str)

    # 如果标签列表不为空，则保存标签文件
    if label_list is not None:
        label_path = os.path.join(
            save_dir, os.path.basename(label_path).replace(".json", ".txt")
        )
        with open(label_path, "w") as txt_file:
            for i in label_list:
                txt_file.writelines(i)
        if not os.path.exists(save_dir + "_preview"):
            os.makedirs(save_dir + "_preview")
        obb_label_to_img(
            img_path,
            label_path,
            os.path.join(
                save_dir + "_preview",
                os.path.basename(label_path).replace(".txt", ".png"),
            ),
        )
        return True
    else:
        return False


def obb_label_to_img(img_path, label_path, save_path):
    """
    将旋转框标签映射到图像上。

    读取图像和标签文件，根据标签信息在图像上绘制旋转框，并保存结果图像。

    参数:
    img_path (str): 图像文件路径。
    label_path (str): 标签文件路径，每行包含一个旋转框的信息。
    save_path (str): 绘制旋转框后图像的保存路径。
    """
    # 读取图像
    img = cv2.imread(img_path)
    # 获取图像的高、宽
    height, width, _ = img.shape
    # 打开标签文件
    with open(label_path, "r") as file_handle:
        # 逐行读取标签信息
        cnt_info = file_handle.readlines()
        # 解析标签信息，去除换行符并分割
        new_cnt_info = [line_str.replace("\n", "").split(" ") for line_str in cnt_info]

    # 遍历每个标签信息
    for new_info in new_cnt_info:
        points = []
        # 提取坐标点，转换为图像坐标
        for i in range(1, len(new_info), 2):
            x_y = [float(tmp) for tmp in new_info[i : i + 2]]
            points.append([int(x_y[0] * width), int(x_y[1] * height)])
        # 在图像上绘制多边形
        cv2.polylines(img, [np.array(points, np.int32)], True, (0, 255, 255))
        # 在图像上标注类别ID
        cv2.putText(
            img,
            class_names[int(new_info[0])],
            (points[0][0], points[0][1] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (36, 255, 12),
            2,
        )
    # 保存绘制后的图像
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.imsave(save_path, img_rgb)


if __name__ == "__main__":
    path_img = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "dataset", "original"
    )

    label_save_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "dataset", "labels"
    )

    if os.path.exists(label_save_path):
        shutil.rmtree(label_save_path)
    os.makedirs(label_save_path)
    for j in tqdm(os.listdir(path_img)):
        if j.endswith(".json"):
            continue
        else:
            if_copy = labelme_to_obb(
                os.path.join(path_img, j),
                os.path.join(path_img, j)[:-4] + ".json",
                label_save_path,
            )

    # 随机划分训练集和验证集
    train_ratio = 0.8
    all_images = [
        f for f in os.listdir(path_img) if f.endswith((".png", ".jpg", ".jpeg"))
    ]
    np.random.shuffle(all_images)
    num_train = int(len(all_images) * train_ratio)
    train_images = all_images[:num_train]
    val_images = all_images[num_train:]
    train_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "dataset", "images", "train"
    )
    val_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "dataset", "images", "val"
    )
    if os.path.exists(train_dir):
        shutil.rmtree(train_dir)
    if os.path.exists(val_dir):
        shutil.rmtree(val_dir)
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    for img in train_images:
        shutil.copy(os.path.join(path_img, img), os.path.join(train_dir, img))
    for img in val_images:
        shutil.copy(os.path.join(path_img, img), os.path.join(val_dir, img))

    # 复制标签文件
    train_label_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "dataset", "labels", "train"
    )
    val_label_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "dataset", "labels", "val"
    )
    if os.path.exists(train_label_dir):
        shutil.rmtree(train_label_dir)
    if os.path.exists(val_label_dir):
        shutil.rmtree(val_label_dir)
    os.makedirs(train_label_dir, exist_ok=True)
    os.makedirs(val_label_dir, exist_ok=True)
    for img in train_images:
        label_file = img.rsplit(".", 1)[0] + ".txt"
        shutil.move(
            os.path.join(label_save_path, label_file),
            os.path.join(train_label_dir, label_file),
        )
    for img in val_images:
        label_file = img.rsplit(".", 1)[0] + ".txt"
        shutil.move(
            os.path.join(label_save_path, label_file),
            os.path.join(val_label_dir, label_file),
        )
    print("数据集划分完成！")
