# ROBOARM 机械臂+深度相机项目

## 机械臂坐标定义

以最下面的舵机为原点，红色为X轴，绿色为Y轴，蓝色为Z轴。

![alt text](docs/image1.png)

## 机械臂标定

见[Roboarm机械臂文档](https://s1vvxephwhf.feishu.cn/wiki/Ulmsw4FPziq8oFkb8eXcx875nfe)

## 步骤

### 数据收集

#### 物品分类

直接拍摄物品图片，标注物品种类训练YOLO模型。

#### 中国象棋

先用相机拍摄包含棋盘的图片，标注棋子种类和位置。训练好YOLO模型后，识别棋盘时根据角点先矫正棋盘，再输入YOLO模型识别，以便定位棋子在棋盘中的位置。
