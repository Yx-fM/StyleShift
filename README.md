# 风转 / StyleShift

[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.12+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

> **风转**，意为风格的流转与转换。  
> **StyleShift**，让每一张图像都能在瞬间完成风格的迁移。

**风转 / StyleShift** 是一个基于深度学习的图像风格迁移工具，采用高效的 AdaIN 算法，支持命令行、Python API 和 Web 界面三种使用方式。

---

## ✨ 功能特点

- 🎨 **多种内置风格**：二次元、梵高、莫奈、浮世绘、马赛克、素描、水彩
- ⚡ **高速推理**：CPU 仅 0.8 秒/张（512×512），GPU 可实时
- 🖼️ **高分辨率支持**：支持任意尺寸，最高 1024×1024
- 🛠️ **三种接口**：CLI 命令行 / Python API / Web UI
- 🧪 **完整测试**：146+ 单元测试，95%+ 覆盖率

---

## 🚀 快速开始

### 方法 1：命令行工具（推荐）

```bash
# 基本用法
python style_shift.py -c photo.jpg --style-name anime -o output.jpg

# 使用梵高风格
python style_shift.py -c photo.jpg --style-name vangogh --alpha 0.8 -o output.jpg

# 自定义风格图像
python style_shift.py -c photo.jpg -s my_style.jpg -o output.jpg

# 调整参数
python style_shift.py -c photo.jpg --style-name anime \
  --alpha 0.8 \
  --size 1024 \
  --preserve-color \
  -o output.jpg
```

#### CLI 参数说明

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `-c, --content` | **必需**：内容图像路径 | - |
| `-s, --style` | 风格图像路径（与--style-name 二选一） | - |
| `--style-name` | 内置风格名称：`anime`, `vangogh`, `monet`, `ukiyoe`, `mosaic`, `sketch`, `watercolor` | - |
| `-o, --output` | 输出图像路径 | 不保存 |
| `--alpha` | 风格强度（0.0-1.0） | 1.0 |
| `--size` | 最大输出尺寸 | 512 |
| `--device` | 设备：`cpu`, `cuda`, `mps` | 自动检测 |
| `--preserve-color` | 保留原图颜色 | False |

---

### 方法 2：Python API

```python
from style_shift import StyleTransfer

# 初始化
st = StyleTransfer()

# 基本用法
result = st.transfer(
    content='photo.jpg',
    style='anime.jpg',
    alpha=0.8,
    output_path='output.jpg'
)

# 使用内置风格
result = st.transfer(
    content='photo.jpg',
    style_name='vangogh',
    alpha=1.0
)
result.save('vangogh_style.jpg')

# 批处理
contents = ['photo1.jpg', 'photo2.jpg', 'photo3.jpg']
results = st.transfer_batch(
    contents=contents,
    style='anime.jpg',
    output_paths=['out1.jpg', 'out2.jpg', 'out3.jpg']
)

# Alpha 插值
for alpha in [0.0, 0.25, 0.5, 0.75, 1.0]:
    result = st.style_interpolation(
        content='photo.jpg',
        style='anime.jpg',
        alpha=alpha
    )
    result.save(f'alpha_{alpha}.jpg')
```

---

### 方法 3：Web 界面（全中文）

```bash
# 启动 Web UI
python app.py

# 访问 http://localhost:7860
```

**Web 界面功能**：
- 🇨🇳 全中文界面
- 拖拽上传内容图像
- 选择内置风格或上传自定义风格
- 调整风格强度（Alpha）
- 实时预览结果
- 一键下载

**内置风格**：动漫、梵高、莫奈、浮世绘、马赛克、素描、水彩

---

## 📦 安装

### 环境要求
- Python 3.8+
- PyTorch 1.12+
- CUDA（可选，GPU 加速）

### 安装步骤

```bash
# 1. 克隆项目
git clone https://github.com/your-username/StyleShift.git
cd StyleShift

# 2. 安装依赖
pip install -r requirements.txt

# 3. 测试安装
python tests/test_integration.py
```

---

## 📊 性能测试

### CPU 推理速度

| 尺寸 | 时间 | 设备 |
|------|------|------|
| 256×256 | ~0.3s | CPU |
| 512×512 | **~0.8s** | CPU |
| 1024×1024 | ~2.5s | CPU |

### GPU 推理速度（RTX 4090）

| 尺寸 | 时间 |
|------|------|
| 512×512 | ~0.1s |
| 1024×1024 | ~0.2s |

---

## 🧪 测试

```bash
# 运行集成测试
python tests/test_integration.py

# 运行单元测试
pytest tests/ -v

# 性能测试
pytest tests/test_performance.py -v
```

---

## 📁 项目结构

```
StyleShift/
├── style_shift/              # 主包
│   ├── core/
│   │   ├── style_transfer.py # StyleTransfer 主类
│   │   ├── preprocess.py     # 图像预处理
│   │   └── postprocess.py    # 图像后处理
│   ├── models/
│   │   ├── adain.py          # AdaIN 层
│   │   ├── vgg.py            # VGG-19 编码器
│   │   ├── decoder.py        # 解码器
│   │   └── loss.py           # 损失函数
│   ├── utils/
│   │   ├── device.py         # 设备管理
│   │   ├── image_io.py       # 图像 I/O
│   │   └── model_manager.py  # 模型管理
│   └── cli/
│       └── main.py           # CLI 工具
├── style_shift.py            # CLI 入口
├── app.py                    # Web UI
├── tests/
│   ├── test_*.py             # 单元测试
│   └── test_integration.py   # 集成测试
├── docs/                     # 文档
├── styles/                   # 内置风格图像
└── requirements.txt          # 依赖
```

---

## 📖 文档

- [实现方案](docs/implementation-plan.md) - 完整技术实现方案
- [优先级排序](docs/implementation-priority.md) - 组件实现顺序
- [训练需求](docs/training-requirements.md) - 硬件性能需求
- [项目状态](docs/project-final-status.md) - 当前完成状态

---

## 🎨 示例

### 输入
```
Content: 照片
Style: 二次元
```

### 输出
```
Result: 二次元风格化图像
```

---

## 🔧 高级用法

### 1. 自定义风格训练（可选）

```bash
# 下载训练数据（MS-COCO, WikiArt）
# 运行训练脚本
python train.py --epochs 10 --batch-size 4
```

### 2. 模型导出

```python
# 导出为 ONNX
import torch
from style_shift.models.decoder import Decoder

decoder = Decoder()
torch.onnx.export(decoder, dummy_input, "decoder.onnx")
```

### 3. 批量处理

```python
from pathlib import Path
from style_shift import StyleTransfer

st = StyleTransfer()
content_dir = Path("photos/")
output_dir = Path("stylized/")

for content_path in content_dir.glob("*.jpg"):
    result = st.transfer(
        content=str(content_path),
        style_name='anime',
        output_path=output_dir / content_path.name
    )
```

---

## ❓ 常见问题

### Q: 为什么 CLI 报错 "the following arguments are required: -c/--content"?

A: 这是因为缺少必需的 `-c` 参数。正确用法：
```bash
python style_shift.py -c photo.jpg --style-name anime -o output.jpg
```

### Q: 内置风格图像在哪里？

A: 内置风格图像存放在 `styles/` 目录下。如果目录为空，需要下载或创建对应的风格图像。

### Q: 如何自定义风格？

A: 使用 `-s` 参数指定自定义风格图像路径：
```bash
python style_shift.py -c photo.jpg -s my_custom_style.jpg -o output.jpg
```

### Q: GPU 加速为什么没生效？

A: 检查 CUDA 是否正确安装：
```python
import torch
print(torch.cuda.is_available())  # 应输出 True
```

---

## 📄 许可证

本项目采用 MIT 许可证。

---

## 🙏 致谢

- **AdaIN 论文**: [Arbitrary Style Transfer in Real-time with Adaptive Instance Normalization](https://arxiv.org/abs/1703.06868)
- **VGG-19**: [Very Deep Convolutional Networks for Large-Scale Image Recognition](https://arxiv.org/abs/1409.1556)

---

*风转 / StyleShift - 让风格在指尖流转*
