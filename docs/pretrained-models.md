# StyleShift 预训练模型使用指南

**日期**: 2026 年 3 月 26 日

---

## 📦 模型大小

### 模型规格

| 属性 | 值 |
|------|-----|
| **参数量** | 3.5M (3,505,219) |
| **架构** | Decoder (转置卷积网络) |
| **输入** | 512×512 特征图 |
| **输出** | 512×512 RGB 图像 |

### 文件大小

| 格式 | 大小 | 说明 |
|------|------|------|
| **FP32** | ~13-15 MB | 标准精度 |
| **FP16** | ~7-8 MB | 半精度压缩 |
| **官方 AdaIN** | ~50 MB | 包含多个风格 |

---

## 🎯 可用模型

### 1. 已训练模型（当前可用）

**位置**: `checkpoints/decoder_final.pth`

**训练信息**:
- 训练轮数：10 epochs
- 批次大小：4
- 数据集：样本数据（40 张图）
- 文件大小：~13 MB

**效果**: 
- ⚠️ 能正常输出图像
- ⚠️ 风格迁移效果有限（样本数据训练）
- ✅ 系统功能验证通过

**使用方法**:
```python
import torch
from style_shift.models.decoder import Decoder

decoder = Decoder()
checkpoint = torch.load('checkpoints/decoder_final.pth', map_location='cpu')
decoder.load_state_dict(checkpoint['decoder_state_dict'])
```

---

### 2. 官方 AdaIN 预训练模型（推荐）

**来源**: [AdaIN 官方 GitHub](https://github.com/xunhuang1995/AdaIN)

**模型信息**:
- 训练数据集：MS-COCO + WikiArt (160K 图像)
- 训练轮数：160K iterations
- 文件大小：~50 MB

**下载方法**:
```bash
# 从官方仓库下载
# 访问：https://github.com/xunhuang1995/AdaIN
# 下载 decoder.pth 文件
```

**注意**: 官方模型架构可能需要微调以匹配当前代码。

---

### 3. 自己训练（最佳效果）

**训练步骤**:

1. **下载数据集** (40GB):
```bash
python scripts/download_datasets.py
```

2. **开始训练** (GPU 1-2 天):
```bash
python train.py --epochs 16 --batch-size 8
```

3. **获得模型**:
```
checkpoints/decoder_final.pth
```

**预计效果**:
- ✅ 明显的风格特征
- ✅ 保留内容结构
- ✅ 无伪影

---

## 🚀 快速开始

### 使用当前训练模型

```python
from style_shift import StyleTransfer

# 初始化（自动使用检查点）
st = StyleTransfer()

# 测试
from PIL import Image
content = Image.new('RGB', (256, 256), 'red')
style = Image.new('RGB', (256, 256), 'blue')

result = st.transfer(content=content, style=style, alpha=0.8)
result.save('output.png')
print(f"Saved: output.png")
```

---

## 📊 模型对比

| 模型 | 大小 | 训练数据 | 效果 | 可用性 |
|------|------|---------|------|--------|
| **当前训练** | 13 MB | 40 张样本 | ⭐⭐ | ✅ 立即可用 |
| **官方 AdaIN** | 50 MB | 160K 图像 | ⭐⭐⭐⭐⭐ | ⚠️ 需架构适配 |
| **自训练** | 13 MB | 160K 图像 | ⭐⭐⭐⭐⭐ | ⏳ 需 1-2 天 |

---

## 💡 建议

### 方案 A: 快速测试（推荐）

**使用当前训练模型**
- ✅ 立即可用
- ✅ 验证系统功能
- ⚠️ 效果有限

**时间**: 5 分钟

---

### 方案 B: 最佳效果

**下载官方模型或自己训练**
- ✅ 效果优秀
- ⏳ 需要时间
- ⚠️ 可能需要架构调整

**时间**: 
- 下载官方：1-2 分钟 + 适配时间
- 自己训练：1-2 天（GPU）

---

## 📝 总结

**当前状态**:
- ✅ 训练系统 100% 完成
- ✅ 已训练模型可用（样本数据）
- ✅ 系统功能验证通过

**下一步**:
1. 使用当前模型测试系统
2. 决定是否需要更好的效果
3. 选择：下载官方模型 或 自己训练

**预训练模型大小**: **~13-15 MB**（很小，易于分发）

---

*文档生成时间：2026 年 3 月 26 日*
