# StyleShift 训练系统说明

**日期**: 2026 年 3 月 26 日  
**状态**: ✅ **训练系统框架已创建**

---

## ⚠️ 重要说明

### 当前 Decoder 状态
**问题**: Decoder 是随机初始化的，没有训练过  
**症状**: 风格迁移后图像变灰，没有风格效果  
**原因**: 需要训练才能产生良好的效果

---

## ✅ 解决方案

### 方案 1：使用预训练模型（立即可用）

**推荐**: 从官方 AdaIN 项目下载预训练 Decoder

```bash
# 下载官方预训练模型
python scripts/download_models.py
```

**效果**: 立即可用，良好的风格迁移效果

---

### 方案 2：自己训练（最佳效果）

**训练系统已创建**:
- ✅ `config/training_config.py` - 训练配置类
- ✅ `config/default_training.yaml` - 默认超参数
- ✅ `train.py` - 主训练脚本

**需要的数据集**:
- MS-COCO (内容图像，80K，~25GB)
- WikiArt (风格图像，80K，~15GB)

**训练时间**:
- GPU (RTX 4090): 1-2 天 (16 epochs)
- CPU: 13-20 天 (不推荐)

**训练命令**:
```bash
python train.py --config config/default_training.yaml --epochs 16
```

---

## 📊 训练配置（效果最优）

| 参数 | 值 | 说明 |
|------|-----|------|
| Epochs | 16 | 官方 AdaIN 使用 160K iterations |
| Batch Size | 8 | 质量和速度的平衡 |
| Learning Rate | 1e-4 | 官方 AdaIN 值 |
| LR Decay | 5e-5 | 每 iteration 衰减 |
| Style Weight | 10.0 | 风格权重（内容:风格 = 1:10） |
| Content Weight | 1.0 | 内容权重 |
| TV Weight | 1e-6 | 平滑正则化 |

---

## 🔧 快速测试（使用随机数据）

```bash
# 测试训练系统是否工作
python train.py --epochs 1 --batch-size 2
```

**注意**: 这只会运行随机数据，不会产生有效模型。需要真实数据集才能训练出好效果。

---

## 📈 训练进度监控

**TensorBoard**:
```bash
tensorboard --logdir runs/
# 访问 http://localhost:6006
```

**查看指标**:
- Loss/Total - 总损失
- Loss/Content - 内容损失
- Loss/Style - 风格损失
- Images - 风格迁移示例

---

## 🎯 验证训练效果

训练完成后，使用 Web UI 测试：

```bash
python app.py
# 访问 http://localhost:7860
```

**好的效果应该**:
- ✅ 明显的风格特征（笔触、颜色）
- ✅ 保留内容结构
- ✅ 无明显伪影

**差的效果**:
- ❌ 图像变灰
- ❌ 颜色失真
- ❌ 没有风格特征

---

## 💡 建议

**立即使用**: 下载预训练模型  
**追求最佳**: 自己训练（1-2 天）  
**测试系统**: 运行快速测试（5 分钟）

---

*文档生成时间：2026 年 3 月 26 日*
