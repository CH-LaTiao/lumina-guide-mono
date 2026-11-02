<p align="center">
  <strong>LUMINA Guide Mono</strong><br>
  单目摄像头版的可穿戴场景理解与辅助通行原型
</p>

LUMINA Guide Mono 是 [LUMINA Guide](https://github.com/CH-LaTiao/lumina-guide)
的独立单目分支。它使用一台普通 UVC 摄像头、NVIDIA Jetson Nano、YOLO11n 与
Depth Anything V2 Small，在同一页面展示实时画面、检测框、单目深度、近似距离、
GPU 状态和建议提示。

> 这是工程验证原型。近似距离不是安全级测距，系统不能替代白手杖、导盲犬、
> 陪同人员或交通判断。默认 `movement_allowed=false`。

## 当前能力

- MJPG/UVC 摄像头采集，支持自动重连、最新帧覆盖和 1–30 FPS 可观测状态。
- YOLO11n ONNX 目标检测；框内显示类别、置信度、近似距离和误差范围。
- Depth Anything V2 Small TensorRT 单目相对深度；支持已知距离标定和目标动态锚定。
- 检测物体使用目标框高度和分类尺寸先验估算米数；镜像、裁切或边界不完整的目标
  会拒绝测距。
- 用户可用已知距离完成单目深度标定；标定后输出连续近似距离与误差范围。
- 站立、步行和暂停模式。步行提示保持一句，优先给出“停”或“向哪走”。
- 浏览器中实时查看画面、检测框、深度图、距离、延迟、帧龄和方向建议。
- 所有移动许可保持关闭；单目结果只作为观察与辅助提示。

## 架构

```text
UVC Camera
  -> capture thread (latest-frame wins)
  -> inference worker
       -> YOLO11n ONNX
       -> class-prior distance
       -> Depth Anything V2 Small TensorRT/CUDA
       -> relative depth / calibrated distance
  -> deterministic guidance
  -> dashboard, JPEG stream and JSON status
```

检测和深度估计运行在独立工作线程，HTTP 页面不会直接占用相机。目标框只在证据
仍新鲜且方向明确时保留；没有可验证的通路时，不输出“可安全通行”。

## 单目测距说明

项目同时使用两种近似测距：

1. **目标先验测距**：`距离 ≈ 实际高度 × 焦距 / 目标框像素高度`。类别高度来自
   通用尺寸表，结果附带较宽误差范围，并标记为 `class_prior`。
2. **单目深度测距**：Depth Anything V2 输出稠密相对逆深度；使用已知距离标定，
   再按 `距离 ≈ 标定值 / 相对逆深度` 计算。未标定时只显示近远排序。

玻璃、镜面、纯色墙面、细杆、地面纹理、镜头俯仰及人物姿态都会影响单目测距。
必须用卷尺、激光测距仪或受控测试集另行验收。

## 快速开始

```bash
cd ~/lumina-guide-mono
./scripts/download_models.sh
./run.sh --host 0.0.0.0 --port 8766
```

浏览器打开 `http://<JETSON_IP>:8766/`。页面无需访问码，但默认只适合可信局域网。

若以下两个同版本 TensorRT 引擎存在，启动器会自动优先使用 CUDA：

```text
models/yolo11n_640_fp16.engine
models/dav2s_518_fp16.engine
```

浏览器顶部会显示实际检测后端、深度后端、GPU 利用率、GPU 温度和物理内存。
缺少引擎时才回退到 ONNX Runtime 或 OpenCV CPU。

### 安装为自动恢复服务

在目标路径固定为 `/home/ubuntu/lumina-guide-mono` 的 Jetson 上：

```bash
sudo install -m 0644 deployment/lumina-guide-mono.service \
  /etc/systemd/system/lumina-guide-mono.service
sudo systemctl daemon-reload
sudo systemctl enable --now lumina-guide-mono.service
sudo systemctl status lumina-guide-mono.service
```

查看实时终端日志：

```bash
journalctl -u lumina-guide-mono.service -f
```

摄像头设备、分辨率、模型路径和水平视场角可通过参数或环境变量调整：

首次安装可创建隔离环境。请保持 `numpy<2` 以匹配系统 OpenCV ABI。TensorRT
引擎由同版本 JetPack/TensorRT 在目标 GPU 上生成；模型编译失败时仍可回退 CPU。

```bash
sudo apt install python3.10-venv
python3 -m venv --system-site-packages .venv
.venv/bin/pip install -r requirements-jetson.txt
```

```bash
CAMERA_DEVICE=0 CAMERA_WIDTH=1280 CAMERA_HEIGHT=720 \
YOLO_MODEL=models/yolo11n.onnx DEPTH_MODEL=models/midas-small.onnx \
CUDA_VISIBLE_DEVICES=0 \
CAMERA_HFOV_DEG=68 ./run.sh --host 0.0.0.0 --port 8766
```

`CAMERA_HFOV_DEG` 应尽量填写摄像头厂商标称值。它直接影响分类先验测距。

### 已知距离标定

把一个清晰物体放在摄像头正前方，测量其到镜头表面的距离，然后在页面“深度标定”
区域输入米数并点击“记录标定”。程序只保存一个比例系数到 `runtime/`，不上传图片。
标定后深度图对应约 `0.4–8 m` 的显示范围；超出范围仍会被标记为不确定。

## 运行状态

- `/`：实时监控页面
- `/status`：机器可读 JSON
- `/stream.mjpg`：带框的实时标注流
- `/depth.jpg`：相对深度热图
- `POST /api/mode`：`standing`、`walking` 或 `paused`
- `POST /api/calibrate`：提交已知距离的深度标定

## 测试

```bash
python3 -m pip install -r requirements-test.txt
python3 -m unittest discover -s tests -v
python3 tools/check_public_secrets.py
node --check tools/static/app.js
```

单元测试使用合成数据，不需要摄像头。实机检查另行记录，不能由无硬件测试替代。

