"""
[Splatting Physical Scenes: End-to-End Real-to-Sim from Imperfect Robot Data](https://arxiv.org/html/2506.04120v2)


sim/camera_pose_opt.py

相机位姿微调（骨架实现）

功能概要：
- 从 mjmodel.xml 中读取名为 `gripper_cam` 的摄像机初始定义（pos, xyaxes）并构造初始位姿 T_init
- 提供可微分渲染器接口占位（支持快速可微分 DummyRenderer）
- 使用 se(3) 6D 参数化（李代数）对相机位姿进行优化，使渲染图像与真实图像的光度误差最小化
- 支持将优化后的位姿写回 mjmodel.xml 中摄像机定义

说明：这是一个可运行的骨架。真实场景中建议替换 `DifferentiableRenderer` 为真实的 PyTorch3D / nvdiffrast / Mitsuba2 渲染器实现。



1. 初始化：首先，根据真实场景的粗略测量，在仿真环境中为相机设置一个初始的、近似的位姿 `T_init`。
2. 渲染与比较：利用仿真器（特别是那些基于3D高斯溅射 (3DGS) 等可微分渲染技术的平台）从当前相机位姿 `T_init` 渲染出一张图像 `I_render`。然后，计算这张渲染图像与对应的真实世界图像 `I_real` 之间的光度误差（Photometric Error），即像素级别的亮度差异。
3. 梯度优化：由于渲染过程是可微分的，系统可以计算出光度误差相对于相机位姿参数的梯度。简单来说，就是计算出“相机应该朝哪个方向移动和旋转，才能让渲染图像更像真实图像”。
4. 迭代更新：使用梯度下降等优化算法，根据计算出的梯度来更新相机的位姿。
5. 收敛：重复执行“渲染-比较-优化”的步骤，直到光度误差不再显著下降，此时得到的相机位姿 `T_cam` 即为最优解，实现了与真实世界的精确对齐。

根据该思路对模拟机械臂上相机进行微调
参数路径为
urdf\meshes\mjmodel.xml-L60
<camera name="gripper_cam" pos="0 0.08 0" xyaxes="1 0 0 0 0.8 -0.6"/>


TODO 更好的话是使用 模型 参数包括像素尺寸、焦距等
参数	值	说明	
`name`	`"gripper_cam"`	相机名称，用于在代码中引用该相机	
`pos`	`"0 0.08 0"`	相机位置 (x, y, z)，相对于父body的局部坐标	
`xyaxes`	`"1 0 0 0 0.8 -0.6"`	相机朝向，定义相机的X轴和Y轴方向	
`mode`	枚举	跟踪模式：`fixed`/`track`/`trackcom`/`targetbody`/`targetbodycom`	
`target`	字符串	目标body名称（用于跟踪模式）	
`fovy`	数值	垂直视场角（度），默认45	
`resolution`	"宽 高"	分辨率，如 `"640 480"`	
`quat`	"x y z w"	用四元数替代xyaxes定义朝向	

参数	类型	默认值	说明	
`fovy`	real	"45"	垂直视场角（度）。透视投影下以角度表示，正交投影下以长度单位表示	
`focal`	real(2)	"0 0"	物理焦距长度（fx, fy），单位与模型一致。与 `fovy` 互斥	
`focalpixel`	int(2)	"0 0"	像素单位焦距（fx, fy）。若同时指定 `focal`，此项优先	
`sensorsize`	real(2)	"0 0"	传感器尺寸（宽 高）。指定后激活所有内参属性，`fovy` 被忽略	
`principal`	real(2)	"0 0"	主点坐标（cx, cy），物理长度单位，相对于图像中心	
`principalpixel`	real(2)	"0 0"	主点坐标（cx, cy），像素单位。若同时指定 `principal`，此项优先	
`resolution`	int(2)	"1 1"	相机分辨率 `[宽 高]`（像素）。注意：此值仅作为元数据保存，实际渲染尺寸由渲染上下文决定	
`output`	枚举	"rgb"	支持的输出类型：`rgb`, `depth`, `distance`, `normal`, `segmentation`。可组合多个，如 `"rgb normal"`	


"""
from __future__ import annotations

import argparse
import os
import xml.etree.ElementTree as ET
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from PIL import Image


def se3_exp(x: torch.Tensor) -> torch.Tensor:
    """将 6D 向量 (omega, v) 映射到 4x4 SE(3) 矩阵。
    x: shape (6,)
    返回：4x4 张量（float32）
    """
    device = x.device
    omega = x[:3]
    v = x[3:]
    theta = torch.norm(omega)
    if theta.item() < 1e-8:
        R = torch.eye(3, device=device)
    else:
        k = omega / theta
        K = torch.tensor([[0, -k[2], k[1]],[k[2], 0, -k[0]],[-k[1], k[0], 0]], device=device)
        R = torch.eye(3, device=device) + torch.sin(theta) * K + (1 - torch.cos(theta)) * (K @ K)
    T = torch.eye(4, device=device)
    T[:3, :3] = R
    T[:3, 3] = v
    return T


def se3_log(T: torch.Tensor) -> torch.Tensor:
    """近似地将 4x4 SE(3) 矩阵映射回 6D se(3) 向量。用于初始化。
    注意：此实现对小旋转较为稳定；对于高精度需求建议使用 Sophus。
    """
    R = T[:3, :3].cpu().numpy()
    t = T[:3, 3].cpu().numpy()
    # rotation -> axis-angle
    theta = np.arccos(max(min((np.trace(R) - 1) / 2.0, 1.0), -1.0))
    if abs(theta) < 1e-8:
        omega = np.zeros(3, dtype=float)
    else:
        rx = (R - R.T) / (2 * np.sin(theta))
        omega = theta * np.array([rx[2,1], rx[0,2], rx[1,0]])
    se3 = np.zeros(6, dtype=float)
    se3[:3] = omega
    se3[3:] = t
    return torch.from_numpy(se3).float()


class DummyDifferentiableRenderer:
    """一个轻量可微渲染器占位：通过对基准图像做连续的亚像素平移来模拟渲染随相机平移变化。

    目的：给出可运行的最小化示例，使得从像素级光度损失到相机平移有梯度流。
    真正部署时请替换为 PyTorch3D / nvdiffrast / Mitsuba2 等真实渲染器。
    """
    def __init__(self, base_img: torch.Tensor, px_scale: float = 100.0):
        """base_img: (C,H,W), float32 [0,1] tensor
        px_scale: 将 world 平移量映射到像素平移比例（仅用于模拟）
        """
        self.base = base_img.unsqueeze(0)  # 1,C,H,W
        self.C, self.H, self.W = base_img.shape
        self.px_scale = px_scale

    def render_image(self, T_cam: torch.Tensor, intrinsics=None) -> torch.Tensor:
        """根据传入的 4x4 相机位姿生成渲染图像（可微）。
        这里仅用 translation.x/y 来生成图像平移，以保持实现简洁且可微分。
        """
        # 提取平移
        tx = T_cam[0, 3]
        ty = T_cam[1, 3]
        # 以像素为单位的偏移（float，可微）
        dx = tx * self.px_scale
        dy = ty * self.px_scale

        # 生成采样网格并平移
        n = 1
        base = self.base.to(T_cam.device)
        N, C, H, W = base.shape[0], base.shape[1], base.shape[2], base.shape[3]
        # 归一化偏移到 [-1,1]
        shift_x = (dx / (W / 2.0)).unsqueeze(0).unsqueeze(2).unsqueeze(3)
        shift_y = (dy / (H / 2.0)).unsqueeze(0).unsqueeze(2).unsqueeze(3)

        # 构建 base 网格
        ys = torch.linspace(-1.0, 1.0, H, device=T_cam.device)
        xs = torch.linspace(-1.0, 1.0, W, device=T_cam.device)
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing='ij')
        grid = torch.stack((grid_x, grid_y), dim=2)  # H,W,2
        grid = grid.unsqueeze(0)  # 1,H,W,2
        grid = grid + torch.cat([shift_x.squeeze(-1).squeeze(-1).permute(0,2,3), shift_y.squeeze(-1).squeeze(-1).permute(0,2,3)], dim=3)
        # grid_sample expects N,H,W,2 with coords in [-1,1]
        out = F.grid_sample(base, grid, mode='bilinear', padding_mode='border', align_corners=True)
        return out.squeeze(0)


class CameraPoseOptimizer(nn.Module):
    """相机位姿优化器（骨架）

    使用 se(3) 6D 参数化（omega, v）优化相机位姿，使渲染图像与真实图像光度差最小。
    """
    def __init__(self, renderer: DummyDifferentiableRenderer, I_real: torch.Tensor,
                 T_init: torch.Tensor, device: str = 'cpu'):
        super().__init__()
        self.device = device
        self.renderer = renderer
        self.I_real = I_real.to(device)
        if T_init.shape == (4, 4):
            se3_init = se3_log(T_init)
        else:
            se3_init = T_init
        self.se3 = nn.Parameter(se3_init.to(device))

    def forward(self) -> torch.Tensor:
        T_cam = se3_exp(self.se3)
        I_render = self.renderer.render_image(T_cam)
        return I_render

    def photometric_loss(self, I_render: torch.Tensor) -> torch.Tensor:
        diff = I_render - self.I_real
        return (diff ** 2).mean()

    def optimize(self, iters: int = 200, lr: float = 1e-2, verbose: bool = True):
        opt = optim.Adam([self.se3], lr=lr)
        history = []
        for i in range(iters):
            opt.zero_grad()
            I_render = self.forward()
            loss = self.photometric_loss(I_render)
            loss.backward()
            opt.step()
            history.append(loss.item())
            if verbose and (i % 20 == 0 or i == iters - 1):
                print(f"iter {i:04d} loss={loss.item():.6f}")
        T_opt = se3_exp(self.se3.detach())
        return T_opt, history


def parse_mjmodel_camera(xml_path: str, camera_name: str = 'gripper_cam') -> Tuple[np.ndarray, np.ndarray]:
    """从 mjmodel.xml 中解析摄像机 pos 与 xyaxes，返回 (pos, R) 其中 R 为 3x3 旋转矩阵。
    如果未找到返回 (None, None)
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()
    for cam in root.iter('camera'):
        if cam.get('name') == camera_name:
            pos_s = cam.get('pos')
            xy_s = cam.get('xyaxes')
            if pos_s is None or xy_s is None:
                return None, None
            pos = np.fromstring(pos_s, sep=' ')
            xy = np.fromstring(xy_s, sep=' ')
            x_axis = xy[:3]
            y_axis = xy[3:6]
            z_axis = np.cross(x_axis, y_axis)
            R = np.stack([x_axis / np.linalg.norm(x_axis),
                          y_axis / np.linalg.norm(y_axis),
                          z_axis / np.linalg.norm(z_axis)], axis=1)
            return pos, R
    return None, None


def write_mjmodel_camera(xml_path: str, camera_name: str, T_opt: torch.Tensor, out_path: Optional[str] = None):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    T = T_opt.cpu().numpy()
    t = T[:3, 3]
    R = T[:3, :3]
    x_axis = R[:, 0]
    y_axis = R[:, 1]
    for cam in root.iter('camera'):
        if cam.get('name') == camera_name:
            cam.set('pos', ' '.join([f"{v:.6f}" for v in t.tolist()]))
            cam.set('xyaxes', ' '.join([f"{v:.6f}" for v in np.concatenate([x_axis, y_axis]).tolist()]))
            break
    if out_path is None:
        out_path = xml_path
    tree.write(out_path)


def load_image_tensor(path: str, device: str = 'cpu') -> torch.Tensor:
    img = Image.open(path).convert('RGB')
    arr = np.array(img).astype(np.float32) / 255.0
    t = torch.from_numpy(arr).permute(2, 0, 1).float().to(device)
    return t


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mjmodel', default='urdf/meshes/mjmodel.xml')
    parser.add_argument('--cam', default='gripper_cam')
    parser.add_argument('--image', required=True, help='真实世界图像路径')
    parser.add_argument('--iters', type=int, default=300)
    parser.add_argument('--lr', type=float, default=1e-2)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--out-xml', default=None)
    args = parser.parse_args()

    device = args.device
    pos, R = parse_mjmodel_camera(args.mjmodel, camera_name=args.cam)
    if pos is None:
        raise RuntimeError(f'未在 {args.mjmodel} 中找到名为 {args.cam} 的 camera 定义')

    # 构造初始 T_init
    T_init = np.eye(4, dtype=float)
    T_init[:3, :3] = R
    T_init[:3, 3] = pos
    T_init_t = torch.from_numpy(T_init).float()

    I_real = load_image_tensor(args.image, device=device)
    # 基准图像（用于 Dummy 渲染器）使用真实图像的拷贝——仅用于示例
    renderer = DummyDifferentiableRenderer(I_real.cpu())

    optimizer = CameraPoseOptimizer(renderer, I_real, T_init_t, device=device)
    T_opt, history = optimizer.optimize(iters=args.iters, lr=args.lr, verbose=True)

    write_mjmodel_camera(args.mjmodel, args.cam, T_opt, out_path=args.out_xml)
    print('优化完成，已写回 XML。')


if __name__ == '__main__':
    main()
