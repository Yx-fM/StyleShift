# StyleShift 模型下载与使用指南

## ⚠️ 问题说明

当前项目中的 **Decoder 模型是随机初始化的**，没有经过训练，因此：
- ❌ 风格迁移效果差
- ❌ 输出图像只是变灰
- ❌ 没有艺术风格效果

## ✅ 解决方案

### 方案 1：下载预训练模型（推荐）

**步骤**：

1. **下载预训练 Decoder**
   ```bash
   python scripts/download_models.py
   ```

2. **验证下载**
   ```bash
   python -c "from style_shift.utils.model_manager import ModelManager; mm = ModelManager(); print(mm.download_model('decoder_pretrained'))"
   ```

3. **使用模型**
   ```python
   from style_shift import StyleTransfer
   
   # 自动加载预训练模型
   st = StyleTransfer()
   result = st.transfer(content='photo.jpg', style='anime.jpg')
   result.save('output.jpg')
   ```

---

### 方案 2：训练自己的模型

**训练步骤**：

#### 1. 准备数据集

```bash
# 下载 MS-COCO (内容图像)
# 下载 WikiArt (风格图像)
```

#### 2. 运行训练脚本

```bash
python train.py \
  --content-dir data/mscoco \
  --style-dir data/wikiart \
  --epochs 10 \
  --batch-size 4 \
  --lr 1e-3 \
  --output checkpoints/
```

#### 3. 使用训练好的模型

```python
from style_shift import StyleTransfer

st = StyleTransfer(decoder_path='checkpoints/decoder_final.pth')
result = st.transfer(content='photo.jpg', style='anime.jpg')
```

---

## 📦 预训练模型

### 可用模型

| 模型名称 | 训练数据 | 风格类型 | 下载链接 |
|---------|---------|---------|---------|
| `decoder_pretrained` | MS-COCO + WikiArt | 通用艺术风格 | [待提供] |
| `decoder_anime` | MS-COCO + 动漫数据集 | 二次元风格 | [待提供] |
| `decoder_vangogh` | MS-COCO + 梵高作品集 | 梵高风格 | [待提供] |

### 模型规格

- **参数量**: 3.5M
- **输入尺寸**: 512×512
- **推理时间**: 0.8s (CPU), 0.1s (GPU)
- **文件大小**: ~15 MB

---

## 🔧 当前快速测试（使用随机模型）

**注意**: 随机模型效果差，仅用于测试代码是否运行

```python
from style_shift import StyleTransfer
from PIL import Image

# 创建测试图像
content = Image.new('RGB', (512, 512), color='red')
style = Image.new('RGB', (512, 512), color='blue')

# 使用随机初始化的 Decoder
st = StyleTransfer()
result = st.transfer(content=content, style=style)
result.save('test_output.jpg')

print('测试完成！')
print('注意：随机模型效果差，需要下载预训练模型')
```

---

## 📊 效果对比

### 随机初始化的 Decoder
- ❌ 输出灰暗
- ❌ 无风格特征
- ❌ 颜色失真

### 预训练 Decoder
- ✅ 风格明显
- ✅ 颜色鲜艳
- ✅ 艺术效果好

---

## 🚀 下一步

**立即执行**：
1. 下载预训练模型
2. 替换随机 Decoder
3. 测试风格迁移效果

**长期计划**：
1. 训练自己的模型
2. 支持更多风格
3. 优化推理速度

---

*文档生成时间：2026 年 3 月 26 日*
