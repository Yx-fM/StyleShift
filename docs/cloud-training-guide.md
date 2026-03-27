# StyleShift 云端训练完全指南

**日期**: 2026 年 3 月 27 日

---

## 一、云端训练流程概述

```
本地准备 → 上传文件 → 云端配置 → 开始训练 → 下载模型
   ↓           ↓           ↓           ↓           ↓
 代码包     Git/FTP    环境配置    监控进度    检查点下载
```

---

## 二、AutoDL 云平台详细教程（推荐）

### 1. 注册与充值

**网址**: https://www.autodl.com/

**步骤**:
1. 手机号注册账号
2. 实名认证（必需）
3. 充值至少 ¥20（推荐 ¥50）

### 2. 创建实例

**推荐配置**:
```
GPU: RTX 4090 24GB
CPU: 8 核
内存：32 GB
存储：50 GB（系统盘）+ 数据盘
镜像：PyTorch 2.0+ CUDA 11.8
区域：选择离你近的（北京/上海/广州）
```

**预计费用**: ¥2/小时

### 3. 启动实例

创建后等待 3-5 分钟启动，然后：
- 点击"控制台"进入 JupyterLab
- 或 SSH 连接（推荐）

### 4. 上传代码（三种方法）

#### 方法 A: Git 克隆（推荐⭐⭐⭐⭐⭐）

**前提**: 代码已上传到 GitHub/Gitee

```bash
# 在 JupyterLab 或 SSH 终端执行
cd /root/autodl-tmp
git clone https://github.com/your-username/StyleShift.git
cd StyleShift
pip install -r requirements.txt
```

**优点**: 
- 速度快
- 可版本管理
- 方便更新

#### 方法 B: JupyterLab 上传（适合小文件）

**步骤**:
1. 打开 JupyterLab
2. 右键 → Upload Files
3. 选择文件（建议打包成 zip）
4. 解压：`unzip StyleShift.zip`

**优点**: 简单直观  
**缺点**: 大文件慢

#### 方法 C: FTP/SFTP 上传

**使用 FileZilla**:
```
主机：你的实例 IP
端口：自动分配（控制台查看）
用户名：root
密码：控制台查看
远程目录：/root/autodl-tmp/
```

**优点**: 适合大文件  
**缺点**: 配置稍复杂

### 5. 上传数据集

#### 方案 A: 云端直接下载（推荐⭐⭐⭐⭐⭐）

```bash
# MS-COCO
cd /root/autodl-tmp/data
wget http://images.cocodataset.org/zips/train2014.zip
unzip train2014.zip

# WikiArt
wget <wikiart-url>
unzip wikiart.zip
```

**优点**: 
- 速度快（云端带宽大）
- 不占用本地流量
- 一次下载永久使用

#### 方案 B: 本地上传（不推荐）

```bash
# 本地打包
tar -czf datasets.tar.gz data/coco data/wikiart

# 上传（使用 FTP 或 scp）
scp datasets.tar.gz root@your-ip:/root/autodl-tmp/

# 云端解压
tar -xzf datasets.tar.gz
```

**缺点**: 
- 40GB 数据上传慢
- 占用本地带宽
- 容易中断

### 6. 环境配置

```bash
# 进入项目目录
cd /root/autodl-tmp/StyleShift

# 创建虚拟环境（可选）
conda create -n styleshift python=3.10
conda activate styleshift

# 安装依赖
pip install -r requirements.txt

# 验证 GPU
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
```

### 7. 修改训练配置

编辑 `config/production_training.yaml`:
```yaml
content_root: "/root/autodl-tmp/data/coco"
style_root: "/root/autodl-tmp/data/wikiart"
batch_size: 8  # 4090 可用 8-16
image_size: 256
epochs: 16
```

### 8. 开始训练

```bash
# 使用 tmux（防止断线中断）
tmux new -s training

# 开始训练
python train.py --config config/production_training.yaml --epochs 16

# 按 Ctrl+B 然后 D 退出 tmux（后台运行）
```

### 9. 监控训练进度

#### 方法 A: TensorBoard（推荐）

```bash
# 在新终端启动
tensorboard --logdir runs/ --host 0.0.0.0 --port 6006

# AutoDL 会自动映射端口
# 在浏览器访问：http://<实例 IP>:6006
```

#### 方法 B: 查看日志

```bash
# 实时查看训练日志
tail -f training.log

# 查看已训练 epochs
ls -lh checkpoints/production/
```

#### 方法 C: AutoDL 控制面板

- 查看 GPU 使用率
- 查看显存占用
- 查看温度

### 10. 训练完成后

```bash
# 查看生成的模型
ls -lh checkpoints/production/
# 应看到：decoder_epoch_1.pth 到 decoder_epoch_16.pth

# 测试模型
python tests/test_style_transfer.py

# 打包模型
tar -czf trained_models.tar.gz checkpoints/
```

### 11. 下载模型到本地

#### 方法 A: JupyterLab 下载

1. 打开 JupyterLab
2. 找到 `trained_models.tar.gz`
3. 右键 → Download

#### 方法 B: SCP 下载

```bash
# 本地执行
scp root@your-ip:/root/autodl-tmp/StyleShift/trained_models.tar.gz ./
```

#### 方法 C: 上传到云存储

```bash
# 上传到百度网盘（使用 bypy）
pip install bypy
bypy upload trained_models.tar.gz

# 或上传到阿里云 OSS
```

### 12. 关闭实例

**重要**: 训练完成后立即关闭实例以节省费用！

```bash
# 在 AutoDL 控制台
实例 → 关机 → 确认
```

**费用结算**: 按秒计费，关机后停止收费

---

## 三、Google Colab 详细教程

### 1. 访问 Colab

**网址**: https://colab.research.google.com/

**需要**: Google 账号

### 2. 创建 Notebook

```
文件 → 新建 Notebook
```

### 3. 配置 GPU

```
代码执行程序 → 更改运行时类型
  运行时类型：Python 3
  硬件加速器：GPU
  GPU 类型：T4（免费）或 V100（Pro）
```

### 4. 上传代码

#### 方法 A: Git 克隆

```python
!git clone https://github.com/your-username/StyleShift.git
%cd StyleShift
!pip install -r requirements.txt
```

#### 方法 B: 上传文件

```python
from google.colab import files
uploaded = files.upload()  # 选择文件
```

### 5. 挂载 Google Drive（持久化存储）

```python
from google.colab import drive
drive.mount('/content/drive')

# 保存检查点到 Drive
!mkdir -p /content/drive/MyDrive/StyleShift/checkpoints
```

### 6. 下载数据集

```python
# MS-COCO
!wget http://images.cocodataset.org/zips/train2014.zip
!unzip train2014.zip

# WikiArt（需要找到可用链接）
!wget <wikiart-url>
!unzip wikiart.zip
```

### 7. 开始训练

```python
# 在代码单元格中
!python train.py --config config/production_training.yaml --epochs 16
```

### 8. 监控进度

```python
# 使用 TensorBoard
%load_ext tensorboard
%tensorboard --logdir runs/

# 或查看日志
!tail -f training.log
```

### 9. 保存结果

```python
# 保存到 Drive
!cp -r checkpoints/ /content/drive/MyDrive/StyleShift/

# 或下载到本地
from google.colab import files
files.download('trained_models.tar.gz')
```

### 10. Colab 限制

| 类型 | 限制 | 说明 |
|------|------|------|
| **免费** | 12 小时/会话 | 超时需重新连接 |
| **Pro** | 24 小时/会话 | $10/月 |
| **Pro+** | 不限 | $50/月 |
| **每周配额** | 约 70 小时 | 超限需等待 |

---

## 四、阿里云 PAI 详细教程

### 1. 开通服务

**网址**: https://www.aliyun.com/product/pai

**步骤**:
1. 注册阿里云账号
2. 实名认证
3. 开通 PAI-DLC（深度学习）
4. 充值至少¥100

### 2. 创建训练任务

**控制台**: https://dlc.console.aliyun.com/

**配置**:
```
任务类型：训练任务
镜像：PyTorch 1.13
GPU: V100×1
CPU: 8 核
内存：32 GB
存储：100 GB
```

### 3. 上传数据

#### 使用 OSS

```bash
# 本地上传到 OSS
ossutil cp -r data/ oss://your-bucket/data/

# 云端挂载
# 在 PAI 控制台配置 OSS 挂载点
```

### 4. 提交训练

**YAML 配置**:
```yaml
job:
  name: styleshift-training
  image: registry.cn-shanghai.aliyuncs.com/pai/pytorch:1.13
  gpu: 1
  
  command:
    - python
    - train.py
    - --epochs
    - "16"
  
  volumes:
    - oss://your-bucket/data:/data
    - oss://your-bucket/output:/output
```

### 5. 监控与日志

- PAI 控制台 → 任务列表
- 查看实时日志
- 查看 GPU 指标

### 6. 下载结果

```bash
# 从 OSS 下载
ossutil cp -r oss://your-bucket/output/checkpoints/ ./
```

---

## 五、文件上传优化技巧

### 1. 打包压缩

```bash
# 本地打包（减少上传时间）
tar -czf StyleShift.tar.gz \
  style_shift/ \
  config/ \
  train.py \
  requirements.txt

# 云端解压
tar -xzf StyleShift.tar.gz
```

### 2. 增量上传

```bash
# 使用 rsync（仅上传变更）
rsync -avz --exclude '__pycache__' \
  ./StyleShift/ \
  root@your-ip:/root/autodl-tmp/StyleShift/
```

### 3. Git 管理

```bash
# 本地
git add .
git commit -m "Update training config"
git push

# 云端
git pull origin main
```

### 4. 数据集处理

**最佳实践**: 云端直接下载，不要本地上传

```bash
# 使用 aria2c 多线程下载（更快）
aria2c -x 16 -s 16 http://images.cocodataset.org/zips/train2014.zip
```

---

## 六、常见问题解答

### Q1: 上传速度慢怎么办？

**A**:
1. 使用 Git 克隆（最快）
2. 打包压缩后上传
3. 使用 FTP 而非 HTTP
4. 选择离你近的区域

### Q2: 训练中断了怎么办？

**A**:
```bash
# 使用 tmux 防止断线
tmux new -s training
python train.py --resume checkpoints/decoder_epoch_X.pth
```

### Q3: 如何节省云端费用？

**A**:
1. 训练完成后立即关机
2. 使用竞价实例（便宜 50-70%）
3. 选择非高峰时段
4. 使用 Google Colab 免费额度

### Q4: 数据会丢失吗？

**A**:
- 关机不会丢失数据
- 删除实例会丢失数据
- 重要文件保存到云存储（OSS/Drive）

### Q5: 如何多人协作？

**A**:
```bash
# 使用 Git 分支
git checkout -b feature/new-architecture
git push origin feature/new-architecture

# 云端拉取
git fetch origin
git merge origin/feature/new-architecture
```

---

## 七、完整操作清单

### 训练前准备

- [ ] 注册云平台账号
- [ ] 完成实名认证
- [ ] 充值至少¥20
- [ ] 代码上传到 Git（可选但推荐）
- [ ] 打包必要文件

### 云端配置

- [ ] 创建 GPU 实例
- [ ] 等待实例启动
- [ ] 连接 JupyterLab/SSH
- [ ] 克隆/上传代码
- [ ] 安装依赖
- [ ] 验证 GPU 可用

### 数据准备

- [ ] 下载 MS-COCO（云端直接下载）
- [ ] 下载 WikiArt（云端直接下载）
- [ ] 验证数据完整性
- [ ] 修改配置文件路径

### 训练执行

- [ ] 启动 tmux 会话
- [ ] 启动 TensorBoard
- [ ] 开始训练
- [ ] 监控训练进度
- [ ] 处理异常情况

### 训练完成

- [ ] 验证模型文件
- [ ] 运行测试脚本
- [ ] 打包模型
- [ ] 下载到本地/云存储
- [ ] 关闭实例

---

## 八、成本核算示例

### AutoDL RTX 4090 训练一次

```
GPU: RTX 4090 @ ¥2/小时
时间：8 小时
GPU 费用：8 × ¥2 = ¥16

存储：50GB @ ¥0.05/GB/天
时间：1 天
存储费：50 × ¥0.05 = ¥2.5

网络：下行 1GB（下载模型）
费用：约¥0.5

总计：¥19
```

### Google Colab Pro

```
月费：$50（约¥350）
可训练：约 20-30 次
单次成本：¥12-17

优势：不限时，可训练多次
```

### 自建 RTX 3060

```
硬件成本：¥3,000
电费：¥5/次
可训练：>1000 次
单次成本（1000 次）：¥3 + ¥5 = ¥8
```

---

## 九、推荐方案总结

| 需求 | 推荐方案 | 成本 | 时间 |
|------|---------|------|------|
| **偶尔训练** | AutoDL | ¥20/次 | 8 小时 |
| **经常训练** | Colab Pro | ¥350/月 | 不限 |
| **高频训练** | 自建 RTX 3060 | ¥3000 | 2 天/次 |
| **生产环境** | 自建 RTX 4090 | ¥30000 | 8 小时/次 |

---

*文档生成时间：2026 年 3 月 27 日*
