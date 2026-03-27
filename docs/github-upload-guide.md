# GitHub 代码上传完整指南

**项目**: StyleShift  
**日期**: 2026 年 3 月 27 日

---

## 方法 1: Git 命令行（最推荐⭐⭐⭐⭐⭐）

### 步骤 1: 初始化 Git 仓库

打开命令提示符（CMD）或 PowerShell：

```bash
cd Q:\All_Items\DreamProjects\StyleShift
git init
```

看到 `Initialized empty Git repository` 表示成功。

### 步骤 2: 关联远程仓库

在 GitHub 上复制你的项目 URL，格式为：
```
https://github.com/你的用户名/StyleShift.git
```

然后执行：

```bash
git remote add origin https://github.com/你的用户名/StyleShift.git
```

验证是否添加成功：

```bash
git remote -v
```

应该显示：
```
origin  https://github.com/你的用户名/StyleShift.git (fetch)
origin  https://github.com/你的用户名/StyleShift.git (push)
```

### 步骤 3: 创建 .gitignore 文件

在上传前，建议先创建 `.gitignore` 文件，排除不需要的文件。

在项目根目录创建 `.gitignore` 文件，内容如下：

```gitignore
# Python 缓存
__pycache__/
*.py[cod]
*.so
.Python

# 虚拟环境
venv/
env/
.venv/

# IDE 配置
.vscode/
.idea/
*.swp
*.swo

# 训练数据和检查点（文件太大）
data/
checkpoints/*.pth
runs/

# 系统文件
.DS_Store
Thumbs.db

# 日志文件
*.log
training.log

# 临时文件
*.tmp
*.bak
```

### 步骤 4: 添加所有文件

```bash
git add .
```

或者添加特定文件：

```bash
git add style_shift/
git add config/
git add train.py
git add README.md
git add docs/
```

查看状态：

```bash
git status
```

### 步骤 5: 提交更改

```bash
git commit -m "Initial commit: StyleShift project

- Core style transfer implementation
- Training pipeline
- Web UI (Gradio)
- CLI interface
- Documentation"
```

### 步骤 6: 推送到 GitHub

```bash
# 重命名分支为 main
git branch -M main

# 推送（首次需要使用 -u）
git push -u origin main
```

如果是 HTTPS 方式，会提示输入 GitHub 用户名和密码（或 Personal Access Token）。

成功推送后显示：

```
Enumerating objects: XXX, done.
Counting objects: 100% (XXX/XXX), done.
Writing objects: 100% (XXX/XXX), done.
To https://github.com/你的用户名/StyleShift.git
 * [new branch]      main -> main
Branch 'main' set up to track remote branch 'main' from 'origin'.
```

### 后续更新

```bash
# 修改代码后
git add .
git commit -m "Add new feature: XXX"
git push
```

---

## 方法 2: GitHub Desktop（图形界面，适合新手⭐⭐⭐⭐）

### 步骤 1: 下载安装

下载地址：https://desktop.github.com/

安装完成后打开 GitHub Desktop。

### 步骤 2: 登录 GitHub

使用你的 GitHub 账号登录。

### 步骤 3: 添加本地仓库

1. 点击 **File** → **Add Local Repository**
2. 点击 **Choose...** 选择文件夹：`Q:\All_Items\DreamProjects\StyleShift`
3. 点击 **Add Repository**

如果提示 "This directory does not appear to be a git repository"，点击 **Create a repository**。

### 步骤 4: 关联远程仓库

1. 点击右上角 **Publish repository**
2. 选择你已创建的 GitHub 项目：`StyleShift`
3. 保持默认设置：
   - ☑️ Keep code private（如果需要私有）
   - ☑️ Add .gitignore（会自动生成）
   - ☑️ Include README
4. 点击 **Publish Repository**

### 步骤 5: 提交和推送

1. 在左下角 **Summary** 输入提交信息，如 "Initial commit"
2. 可选在 **Description** 输入详细描述
3. 点击 **Commit to main**
4. 点击顶部 **Push origin** 按钮

### 后续更新

1. 修改代码后，GitHub Desktop 会自动检测更改
2. 输入提交信息
3. 点击 **Commit to main**
4. 点击 **Push origin**

---

## 方法 3: VS Code 集成（开发者推荐⭐⭐⭐⭐⭐）

### 步骤 1: 打开项目

1. 打开 VS Code
2. **File** → **Open Folder**
3. 选择：`Q:\All_Items\DreamProjects\StyleShift`

### 步骤 2: 初始化 Git

1. 点击左侧 **Source Control** 图标（或按 `Ctrl+Shift+G`）
2. 点击 **Initialize Repository**

### 步骤 3: 暂存文件

在 Source Control 面板：
1. 点击 **Changes** 上方的 **+** 号（或 **Stage All Changes**）
2. 所有文件移动到 **Staged Changes**

### 步骤 4: 提交

1. 在输入框输入提交信息：
   ```
   Initial commit: StyleShift project
   ```
2. 按 `Ctrl+Enter` 或点击 **Commit** 按钮

### 步骤 5: 关联远程仓库

1. 点击 **...** → **Remote** → **Add Remote**
2. 输入你的 GitHub 项目 URL：
   ```
   https://github.com/你的用户名/StyleShift.git
   ```
3. 按 Enter

### 步骤 6: 推送

1. 点击 **...** → **Push**
2. 首次会提示选择分支，选择 **main**
3. 可能需要登录 GitHub

### 后续更新

1. 修改代码后，Source Control 会显示更改
2. 点击 **+** 暂存
3. 输入提交信息
4. **Commit** → **Push**

或使用快捷键：
- `Ctrl+A`：暂存所有
- `Ctrl+Enter`：提交
- `Ctrl+Shift+P` → "Git: Push"：推送

---

## 推荐的项目结构

上传前确保项目结构清晰：

```
StyleShift/
├── .gitignore              ✅ 必需
├── README.md               ✅ 必需
├── requirements.txt        ✅ 必需
├── pyproject.toml         ✅ 推荐
├── style_shift/           ✅ 核心代码
│   ├── __init__.py
│   ├── core/
│   ├── models/
│   └── utils/
├── config/                ✅ 配置文件
│   └── default_training.yaml
├── train.py               ✅ 训练脚本
├── app.py                 ✅ Web UI
├── style_shift.py         ✅ CLI 入口
├── docs/                  ✅ 文档
│   ├── README.md
│   └── *.md
└── tests/                 ✅ 测试
    ├── __init__.py
    └── test_*.py
```

---

## 大文件处理

### 问题：文件超过 100MB

GitHub 限制单个文件最大 100MB。如果遇到：

```
remote: error: File xxx.pth is 200.00 MB; this exceeds GitHub's file size limit
```

### 解决方案 1: 排除大文件

在 `.gitignore` 中添加：

```gitignore
# 模型检查点（太大）
checkpoints/*.pth

# 原始数据集
data/raw/
datasets/*.zip

# TensorBoard 日志
runs/
```

### 解决方案 2: 使用 Git LFS

```bash
# 安装 Git LFS
git lfs install

# 跟踪大文件
git lfs track "*.pth"
git lfs track "*.zip"

# 提交 .gitattributes
git add .gitattributes
git commit -m "Configure Git LFS"

# 正常提交和推送
git add checkpoints/
git commit -m "Add model checkpoints"
git push
```

### 解决方案 3: 使用云存储

将大文件上传到：
- Google Drive
- 百度网盘
- Hugging Face
- AWS S3

在 README 中提供下载链接。

---

## 常见问题解决

### Q1: 提示 "remote origin already exists"

**原因**: 已经添加过远程仓库

**解决**:
```bash
# 删除旧的
git remote remove origin

# 重新添加
git remote add origin https://github.com/你的用户名/StyleShift.git
```

### Q2: 推送失败 "failed to push some refs"

**原因**: 远程仓库有本地没有的提交

**解决**:
```bash
# 先拉取远程更改
git pull origin main --rebase

# 解决冲突（如果有）

# 再推送
git push -u origin main
```

### Q3: 忘记密码/Token

**解决**:
1. 访问 https://github.com/settings/tokens
2. 创建新的 Personal Access Token
3. 使用 Token 代替密码

或使用 SSH：
```bash
# 生成 SSH 密钥
ssh-keygen -t ed25519 -C "your_email@example.com"

# 添加公钥到 GitHub
# https://github.com/settings/keys

# 修改远程 URL
git remote set-url origin git@github.com:你的用户名/StyleShift.git

# 推送
git push -u origin main
```

### Q4: 文件被意外提交

**解决**:
```bash
# 从 Git 历史中删除文件
git rm --cached path/to/file

# 提交更改
git commit -m "Remove sensitive file"

# 推送
git push
```

### Q5: 想重新提交所有文件

**解决**:
```bash
# 撤销最后一次提交（保留文件）
git reset --soft HEAD~1

# 重新添加
git add .

# 重新提交
git commit -m "New commit message"

# 强制推送（慎用）
git push -f origin main
```

---

## 最佳实践

### 1. 提交信息规范

```bash
# 好的提交信息
git commit -m "feat: Add style transfer optimization"
git commit -m "fix: Fix memory leak in DataLoader"
git commit -m "docs: Update README with installation guide"

# 不好的提交信息
git commit -m "update"  ❌
git commit -m "fix bug"  ❌
git commit -m "aaa"      ❌
```

### 2. .gitignore 最佳实践

```gitignore
# 不要提交的文件
__pycache__/
*.pyc
.env
*.log
checkpoints/*.pth
data/raw/

# 可以提交的文件
requirements.txt
config/*.yaml
docs/*.md
tests/*.py
```

### 3. 分支管理

```bash
# 创建新分支
git checkout -b feature/new-architecture

# 切换分支
git checkout main

# 合并分支
git merge feature/new-architecture

# 删除分支
git branch -d feature/new-architecture
```

### 4. 定期备份

```bash
# 添加多个远程
git remote add backup git@gitlab.com:你的用户名/StyleShift.git

# 推送到所有远程
git push --all
```

---

## 上传后的验证

### 1. 检查 GitHub 页面

访问：https://github.com/你的用户名/StyleShift

确认：
- ✅ 所有文件已显示
- ✅ README 正确渲染
- ✅ 提交历史正确

### 2. 克隆测试

```bash
# 在其他地方克隆
cd /tmp
git clone https://github.com/你的用户名/StyleShift.git

# 验证文件完整
cd StyleShift
ls -la
```

### 3. 检查 .gitignore

确保没有提交不需要的文件：

```bash
# 查看已跟踪的文件
git ls-files

# 不应包含：
# - __pycache__/
# - *.pth
# - data/
```

---

## 总结

| 方法 | 适合人群 | 难度 | 推荐度 |
|------|---------|------|--------|
| **Git 命令行** | 开发者 | ⭐⭐ | ⭐⭐⭐⭐⭐ |
| **GitHub Desktop** | 新手 | ⭐ | ⭐⭐⭐⭐ |
| **VS Code** | VS Code 用户 | ⭐⭐ | ⭐⭐⭐⭐⭐ |

**推荐流程**:
1. 首次上传：GitHub Desktop（简单）
2. 日常更新：VS Code 或命令行
3. 大文件：使用 Git LFS 或云存储

---

*文档生成时间：2026 年 3 月 27 日*
