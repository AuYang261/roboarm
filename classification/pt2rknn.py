
'''
conda create -n rknn python=3.10
conda activate rknn
pip install -r ./calibration/requirements_cp310-2.3.2.txt
pip install ./calibration/rknn_toolkit2-2.3.2-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl
pip install ultralytics
export RKNN_TARGET_PLATFORM=rk3588
/usr/bin/env RKNN_TARGET_PLATFORM=rk3588 /home/hyl/anaconda3/envs/rknn/bin/python /home/hyl/wowrobo/classification/pt2rknn.py
'''

def pt2onnx(model_path: str):
    from ultralytics import YOLO
    import os
    # 加载模型
    model = YOLO(model=model_path)

    # 获取类别名称
    if hasattr(model, 'names') and model.names is not None:
        class_names = model.names
        print(f"Model has {len(class_names)} classes")
        # 保存类别名称到 labels.txt
        labels_path = model_path.replace('.pt', '_labels.txt')
        with open(labels_path, 'w', encoding='utf-8') as f:
            for i in range(len(class_names)):
                f.write(f"{class_names[i]}\n")
        print(f"Labels saved to {labels_path}")
    else:
        print("Warning: Model does not have 'names' attribute")

    # 导出为 ONNX（固定输入尺寸，如 640x640）
    model.export(format="onnx", opset=19, imgsz=640, dynamic=False, simplify=True)
    
def onnx2rknn(onnx_path: str):
    from rknn.api import RKNN
    import os

    rknn = RKNN()

    # 配置模型输入输出
    print('--> Config model')
    target_platform = os.environ.get('RKNN_TARGET_PLATFORM', 'rk3588')
    # 对于YOLO模型，输入为归一化到[0,1]的RGB图像
    # mean和std根据训练时的预处理设置
    # 使用基本配置，量化参数在build时设置
    rknn.config(mean_values=[[0, 0, 0]], std_values=[[255, 255, 255]],
                target_platform=target_platform)

    # 加载 ONNX 模型（需在 config 之后调用）
    print('--> Loading ONNX model')
    ret = rknn.load_onnx(model=onnx_path)
    if ret != 0:
        print('Load ONNX model failed!')
        exit(ret)

    # 编译模型
    print('--> Building RKNN model')
    # 检查是否有量化数据集
    dataset = './calibration/dataset.txt' if os.path.exists('./calibration/dataset.txt') else None

    if dataset:
        # 有数据集，进行量化
        print(f'--> Using dataset for quantization: {dataset}')
        ret = rknn.build(do_quantization=True, dataset=dataset)
    else:
        # 无数据集，不进行量化
        print('--> No quantization dataset found, building without quantization')
        ret = rknn.build(do_quantization=False)

    if ret != 0:
        print('Build RKNN model failed!')
        exit(ret)

    # 导出 RKNN 模型
    rknn_path = onnx_path.replace('.onnx', '.rknn')
    print(f'--> Export RKNN model to {rknn_path}')
    ret = rknn.export_rknn(rknn_path)
    if ret != 0:
        print('Export RKNN model failed!')
        exit(ret)

    # 尝试导出类别信息
    try:
        # 读取模型中的类别信息
        labels_path = onnx_path.replace('.onnx', '_labels.txt')
        if os.path.exists(labels_path):
            with open(labels_path, 'r', encoding='utf-8') as f:
                labels = [line.strip() for line in f.readlines()]
            # 创建包含类别信息的配置文件
            config_path = rknn_path.replace('.rknn', '_config.yaml')
            with open(config_path, 'w', encoding='utf-8') as f:
                f.write(f"classes: {labels}\n")
                f.write(f"num_classes: {len(labels)}\n")
            print(f"Config saved to {config_path}")
    except Exception as e:
        print(f"Warning: Could not save config: {e}")

    print('done!')
    
def main():
    model_path = "./classification/object_detect/runs/best-screw.pt"
    pt2onnx(model_path)
    onnx_path = model_path.replace('.pt', '.onnx')
    onnx2rknn(onnx_path)


if __name__ == "__main__":
    main()