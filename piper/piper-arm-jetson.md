# Plan: Piper SDK 环境安装 & 让机械臂动起来

## Context

目标：在远程主机 `hyl@192.168.0.192`（仅连接 Piper 机械臂）上克隆本仓库，用 uv 管理 Python 环境，安装 piper_sdk，先完成 CAN 连通性检查，再编写测试脚本，使机械臂执行一次关节运动。

---

## Step 1：SSH 登录远程主机，检查硬件与驱动基线

```bash
ssh hyl@192.168.0.192

# 先看系统是否已经识别到 CAN 网络接口
ip link show | grep can
ls /sys/class/net | grep can

# 再看 USB-CAN 是否被 USB 层识别
lsusb | grep -i can

# 检查 gs_usb 驱动是否已加载
lsmod | grep gs_usb
```

说明：
- 如果 `lsusb` 能看到 USB-CAN，但 `ip link show | grep can` 看不到任何 `can*`/`can_piper`，通常是 `gs_usb` 内核驱动未加载，直接跳到文末“故障排查 / bug2”。
- 后续所有命令都不要预设接口一定是 `can0`；必须以实际检测到的接口名为准。

---

## Step 2：安装系统依赖

```bash
sudo apt update && sudo apt install -y can-utils ethtool

# 安装 uv（如果尚未安装）
curl -Lsf https://astral.sh/uv/install.sh | sh
source ~/.bashrc   # 或重开终端使 uv 生效
```

---

## Step 3：克隆仓库并初始化 uv 环境

```bash
git clone -c http.proxy=http://192.168.0.15:7890 https://github.com/AuYang261/roboarm.git ~/roboarm
cd ~/roboarm

uv venv --python 3.8
source .venv/bin/activate

# piper_sdk + python-can 的 gs_usb 支持
uv pip install piper_sdk "python-can[gs_usb]" pyusb

# 验证 piper_sdk 安装
uv run python -c "from piper_sdk import C_PiperInterface_V2; print('piper_sdk OK')"
```

如果无法 clone，http/https 可尝试使用代理 `http://192.168.0.22:7890`。

---

## Step 4：识别真实 CAN 接口，并激活它

先定位 piper_sdk 安装目录，再用 SDK 自带脚本找出“接口名 + USB 硬件地址”的真实映射：

```bash
source .venv/bin/activate

SDK_DIR=$(python - <<'PY'
import inspect, os, piper_sdk
print(os.path.dirname(inspect.getfile(piper_sdk)))
PY
)
echo "$SDK_DIR"

bash "$SDK_DIR/find_all_can_port.sh"
```

示例输出可能类似：

```text
Interface can0 is connected to USB port 81102f0000.mttcan
Interface can1 is connected to USB port 8110300000.mttcan
Interface can2 is connected to USB port 8110330000.mttcan
Interface can3 is connected to USB port 8110340000.mttcan
Interface can_piper is connected to USB port 1-4.2:1.0
```

此时，接入的 USB-CAN 设备对应的是：
- 接口名：`can_piper`
- USB 硬件地址：`1-4.2:1.0`

注意：上面只是示例。**你必须使用自己机器上 `find_all_can_port.sh` 的真实输出，不要手填或猜测。**

### 方法 A：用 SDK 自带脚本激活

```bash
CAN_IF="can_piper"      # 示例；如果实际显示为 can0/can4，就填真实名字
USB_ADDR="1-4.2:1.0"    # 示例；必须替换为真实 USB 地址

sudo ip link set "$CAN_IF" down 2>/dev/null || true
bash "$SDK_DIR/can_activate.sh" "$CAN_IF" 1000000 "$USB_ADDR"
```

### 方法 B：手动激活

```bash
CAN_IF="can_piper"      # 示例；改成真实接口名

sudo ip link set "$CAN_IF" down 2>/dev/null || true
sudo ip link set "$CAN_IF" up type can bitrate 1000000
sudo ip link set "$CAN_IF" txqueuelen 1000
```

验证：

```bash
ip -details link show "$CAN_IF"
```

应看到接口处于 `UP` 状态，波特率为 `1000000`。

---

## Step 5：先做只读连通性测试

在远程创建只读测试脚本 `~/roboarm/test/piper_test_read.py`：

```python
#!/usr/bin/env python3
"""最小连通性测试：只读取 Piper 状态，不发送运动指令。"""
import time
from piper_sdk import C_PiperInterface_V2

can_name = "can_piper"  # 改成 Step 4 确认的真实接口名，如 can0 / can4 / can_piper

piper = C_PiperInterface_V2(can_name)
piper.ConnectPort()
time.sleep(0.1)

print(f"使用 CAN 接口: {can_name}")
print(piper.GetArmJointMsgs())
print(piper.GetArmGripperMsgs())
```

运行：

```bash
uv run python test/piper_test_read.py
```

如果这一步都读不到状态，先不要发送运动指令，优先处理文末故障排查中的 CAN/驱动问题。

---

## Step 6：创建运动测试脚本

在远程创建测试脚本 `~/roboarm/test/piper_test_move.py` 并尝试运行：

```python
#!/usr/bin/env python3
"""最小测试：使能 Piper 机械臂，执行一次关节运动后回零。"""
import time
from piper_sdk import C_PiperInterface_V2

can_name = "can_piper"  # 改成 Step 4 确认的真实接口名，如 can0 / can4 / can_piper
piper = C_PiperInterface_V2(can_name)
piper.ConnectPort()
time.sleep(0.1)

print(f"正在使用接口 {can_name} 使能机械臂...")
while not piper.EnablePiper():
    time.sleep(0.01)
print("使能成功")

factor = 57295.7795  # 1000 * 180 / pi

# 移动到偏移姿态（弧度值）
pos = [0.2, 0.2, -0.2, 0.3, -0.2, 0.5]
joints = [round(p * factor) for p in pos]
piper.MotionCtrl_2(0x01, 0x01, 30, 0x00)  # 速度 30%
piper.JointCtrl(*joints)
print(f"关节指令已发送: {joints}")
time.sleep(3)

# 回零
piper.JointCtrl(0, 0, 0, 0, 0, 0)
print("回零完成")
time.sleep(2)
```

---

## Step 7：在远程主机运行测试

```bash
uv run python test/piper_test_move.py
```

---

## 验证标准

- `uv run python test/piper_test_read.py` 能正常打印关节/夹爪状态
- `EnablePiper()` 成功，打印“使能成功”
- 机械臂可见地移动到偏移姿态后回零
- `ip -details link show <实际接口名>` 显示接口处于 `UP`
- 无明显 CAN 报错（`dmesg | tail -20 | grep -i can`）

---

## 注意事项

- **机械臂处于 slave 模式才能接收控制指令**；若处于 master 模式，需先重启臂体
- CAN 波特率固定为 **1000000**，不要误填成 `100000`
- `judge_flag=False`（默认）适用于第三方 USB-CAN 适配器
- `can_activate.sh` / `find_all_can_port.sh` 都在 `piper_sdk` 安装目录中，可通过 Step 4 的 `SDK_DIR` 动态定位
- 若接口在重新插拔后从 `can_piper` 变成 `can4` 等新名字，脚本中的 `can_name` 也要同步修改

---

## 故障排查 / bug1：`can_activate.sh` 提示找不到 USB 地址对应的 CAN 接口

典型报错：

```text
Error: Unable to find CAN interface corresponding to USB hardware address 1-2:1.0.
```

这通常不是 `can_activate.sh` 本身的问题，而是你传入了错误的 USB 硬件地址，或者接口名并不是你以为的 `can0`。

处理方式：

```bash
bash "$SDK_DIR/find_all_can_port.sh"
```

先拿到真实映射关系，再用真实值重试，例如：

```bash
bash "$SDK_DIR/can_activate.sh" can_piper 1000000 "1-4.2:1.0"
```

要点：
- `1-2:1.0`、`1-4.2:1.0` 这类值是 USB 硬件地址，不是 CAN 接口名
- 系统实际创建的网络接口可能是 `can0`、`can4`，也可能已经被重命名为 `can_piper`
- **不要猜接口名和 USB 地址，必须先查再用**

---

## 故障排查 / bug2：USB-CAN 被识别，但没有生成 CAN 网络接口 / `SEND_MESSAGE_FAILED (100017)`

典型现象：
- `lsusb` 能看到设备，但 `ip link show | grep can` 看不到对应接口
- `EnablePiper()` 或其他发送接口报：

```text
SendCanMessage(SEND_MESSAGE_FAILED (100017))
```

高概率原因：`gs_usb` **内核驱动没有加载**。仅安装 Python 包不够，内核里必须有 `gs_usb` 模块。

先检查：

```bash
lsmod | grep gs_usb
zcat /proc/config.gz | grep GS_USB
```

如果看到类似：

```text
# CONFIG_CAN_GS_USB is not set
```

说明当前内核没有启用 `gs_usb`，需要为当前运行内核补上对应版本的 `gs_usb.ko`。

### Jetson 上的处理思路

1. 安装编译依赖：

```bash
sudo apt update
sudo apt install -y build-essential bc kmod flex bison libncurses-dev libssl-dev dwarves wget git
```

2. 获取**与当前 `uname -r` 完全匹配**的 Jetson 内核源码，在源码目录中：

```bash
zcat /proc/config.gz > .config
sed -i 's/CONFIG_LOCALVERSION=""/CONFIG_LOCALVERSION="-tegra"/' .config
make olddefconfig

export LOCALVERSION=-tegra
export IGNORE_PREEMPT_RT_PRESENCE=1
make -j$(nproc) modules
```

3. 安装 `gs_usb.ko`：

```bash
sudo mkdir -p /lib/modules/$(uname -r)/kernel/drivers/net/can/usb
sudo cp drivers/net/can/usb/gs_usb.ko /lib/modules/$(uname -r)/kernel/drivers/net/can/usb/
sudo depmod -a
sudo modprobe gs_usb
lsmod | grep gs_usb
```

4. 设置开机自动加载：

```bash
echo "gs_usb" | sudo tee /etc/modules-load.d/gs_usb.conf
```

补充说明：
- `uv pip install "python-can[gs_usb]" pyusb` 只是补齐用户态依赖，**不能替代内核模块**
- 如果内核模块未就绪，SDK 仍然可能报 `SEND_MESSAGE_FAILED (100017)`

---

## 故障排查 / bug3：`Message NOT sent` / 发送偶发失败

典型现象：
- CAN 接口存在，但发送时偶发 `Message NOT sent`
- SDK 报 `SEND_MESSAGE_FAILED (100017)`
- 重新插拔后接口名可能变化，例如从 `can_piper` 变成 `can4`

推荐恢复流程：

```bash
CAN_IF="can_piper"   # 如果重新插拔后名字变了，改成新的接口名

# 1. 关闭接口
sudo ip link set "$CAN_IF" down 2>/dev/null || true

# 2. 卸载 gs_usb 驱动
sudo modprobe -r gs_usb

# 3. 物理拔掉 USB-CAN 适配器，等待 3 秒后重新插入
# 4. 将机械臂断电再上电

# 5. 重新加载驱动
sudo modprobe gs_usb

# 6. 等待设备重新出现，并重新确认接口名
sleep 2
ls /sys/class/net | grep can

# 7. 用新的接口名重新配置
CAN_IF="can_piper"   # 这里替换成重新探测后的真实接口名，例如 can4
sudo ip link set "$CAN_IF" type can bitrate 1000000
sudo ip link set "$CAN_IF" txqueuelen 1000
sudo ip link set "$CAN_IF" up
```

恢复后，建议先重新执行只读脚本：

```bash
uv run python test/piper_test_read.py
```

确认收发恢复正常，再执行运动脚本。

---

## 总结

推荐排查顺序：
1. 先确认 `gs_usb` 已加载
2. 再用 `find_all_can_port.sh` 确认真实接口名与 USB 地址
3. 先跑 `piper_test_read.py` 验证连通性
4. 最后再跑 `piper_test_move.py` 发送运动指令
