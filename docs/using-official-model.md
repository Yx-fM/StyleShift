# 官方 AdaIN 预训练模型使用指南

**日期**: 2026 年 3 月 26 日

---

## 📥 下载官方模型

### 方法 1: GitHub 下载（推荐）

1. **访问官方仓库**:
   - GitHub: https://github.com/xunhuang1995/AdaIN
   - 项目主页：https://xunhuang.github.io/AdaIN/

2. **下载模型**:
   ```bash
   # 从 releases 或项目页面下载
   # 文件名：decoder.pth
   # 大小：~50 MB
   ```

3. **保存到项目**:
   ```bash
   mkdir -p models/official
   # 将 decoder.pth 保存到 models/official/
   ```

---

### 方法 2: 直接链接

**Dropbox 链接**:
```
https://www.dropbox.com/s/lq92mw3a74u18p6/decoder.pth?dl=1
```

**Google Drive**:
```
# 需要科学上网
```

---

### 方法 3: 百度网盘（国内）

**链接**: (需要自行搜索资源)
- 搜索关键词："AdaIN 预训练模型 百度网盘"

---

## 🔧 模型适配

官方 AdaIN 模型架构可能与当前代码略有不同，需要适配：

### 官方架构

```python
# 官方 AdaIN Decoder 结构
decoder = nn.Sequential(
    nn.ReflectionPad2d((0, 0, 0, 3)),
    nn.Conv2d(512, 256, kernel_size=3, padding=0),
    nn.ReLU(inplace=True),
    nn.Upsample(scale_factor=2, mode='nearest'),
    # ... 更多层
    nn.Conv2d(3, 3, kernel_size=1),
    nn.Tanh()
)
```

### 当前架构

```python
# 当前 Decoder 结构（见 models/decoder.py）
decoder = Decoder()
```

---

## 🧪 测试官方模型

### 加载并测试

```python
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms

# 1. 加载官方模型
official_decoder = torch.load('models/official/decoder.pth', map_location='cpu')
print(f"Loaded official model")
print(f"  Type: {type(official_decoder)}")

# 如果是 OrderedDict，需要包装成完整模型
if isinstance(official_decoder, dict):
    decoder = Decoder()  # 创建当前 Decoder
    decoder.load_state_dict(official_decoder)
else:
    decoder = official_decoder

decoder.eval()

# 2. 测试
content = Image.open('content.jpg').convert('RGB')
style = Image.open('style.jpg').convert('RGB')

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                        std=[0.229, 0.224, 0.225])
])

content_tensor = transform(content).unsqueeze(0)
style_tensor = transform(style).unsqueeze(0)

# 3. 风格迁移
with torch.no_grad():
    vgg = VGG19Encoder(pretrained=True)
    adain = AdaIN()
    
    content_feat, _ = vgg(content_tensor)
    _, style_feats = vgg(style_tensor)
    style_feat = style_feats['conv4_2']
    
    adain_feat = adain(content_feat, style_feat)
    output = decoder(adain_feat)

# 4. 保存结果
from torchvision.utils import save_image
save_image(output, 'official_result.png', normalize=True)
print("Saved: official_result.png")
```

---

## 📊 效果对比

### 当前训练模型

| 指标 | 值 |
|------|-----|
| 训练数据 | 40 张样本图 |
| 训练轮数 | 10 epochs |
| 输出范围 | [-0.02, 0.05] |
| 效果 | ⭐⭐ 有限 |

### 官方 AdaIN 模型

| 指标 | 值 |
|------|-----|
| 训练数据 | MS-COCO + WikiArt (160K) |
| 训练轮数 | 160K iterations |
| 输出范围 | [0, 1] (normalized) |
| 效果 | ⭐⭐⭐⭐⭐ 优秀 |

---

## 💡 使用建议

### 方案 A: 使用官方模型（推荐）

**优点**:
- ✅ 效果优秀
- ✅ 立即可用（下载后）
- ✅ 经过验证

**缺点**:
- ⚠️ 需要手动下载
- ⚠️ 可能需要架构适配

**步骤**:
1. 下载 decoder.pth (~50 MB)
2. 保存到 `models/official/`
3. 运行测试脚本

---

### 方案 B: 继续训练当前模型

**优点**:
- ✅ 完全兼容当前代码
- ✅ 可自定义训练数据

**缺点**:
- ⏳ 需要 1-2 天训练时间
- ⏳ 需要完整数据集（40GB）

**步骤**:
1. 下载 MS-COCO + WikiArt
2. 运行 `python train.py --epochs 16`
3. 获得自定义模型

---

## 🚀 快速测试

**测试当前模型效果**：

```bash
python << 'EOF'
from style_shift import StyleTransfer
from PIL import Image

st = StyleTransfer()

# 创建测试图像
content = Image.new('RGB', (256, 256), 'red')
style = Image.new('RGB', (256, 256), 'blue')

# 风格迁移
result = st.transfer(content=content, style=style, alpha=0.8)
result.save('test_output.png')
print("测试完成！")
EOF
```

---

## 📝 总结

**当前状态**:
- ✅ 训练系统 100% 完成
- ⚠️ 当前模型效果有限（样本数据训练）
- 🎯 推荐使用官方 AdaIN 模型获得最佳效果

**下一步**:
1. 下载官方模型（50 MB）
2. 测试官方模型效果
3. 对比两种模型的效果差异

**官方模型大小**: **~50 MB**
**下载时间**: 1-5 分钟（取决于网络）

---

*文档生成时间：2026 年 3 月 26 日*
