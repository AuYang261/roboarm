

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
    # 加载模型
    model = YOLO(model=model_path)

    # 导出为 ONNX（固定输入尺寸，如 640x640）
    model.export(format="onnx", opset=19, imgsz=1280, dynamic=False, simplify=True)
    
def onnx2rknn(onnx_path: str):
    from rknn.api import RKNN
    import os

    rknn = RKNN()

    # 配置模型输入输出
    print('--> Config model')
    target_platform = os.environ.get('RKNN_TARGET_PLATFORM', 'rk3588')
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

    print('done!')
    
def main():
    model_path = "./classification/object_detect/runs/best-screw.pt"
    pt2onnx(model_path)
    onnx_path = model_path.replace('.pt', '.onnx')
    onnx2rknn(onnx_path)


if __name__ == "__main__":
    main()