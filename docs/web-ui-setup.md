# Web UI 安装指南

## 问题

运行 `python app.py` 时出现错误：
```
ModuleNotFoundError: No module named 'gradio'
```

## 解决方案

Gradio 是**可选依赖**，仅在需要使用 Web UI 时安装。

### 安装方法

```bash
# 安装 gradio
pip install gradio

# 或者安装完整开发依赖
pip install -r requirements-dev.txt
```

### 验证安装

```bash
python -c "import gradio; print(f'Gradio version: {gradio.__version__}')"
```

应输出：
```
Gradio version: 6.x.x
```

### 启动 Web UI

```bash
python app.py
```

然后访问：**http://localhost:7860**

## 注意事项

- Gradio 6.0+ 可能需要 Python 3.10+
- 如果遇到兼容性问题，可以安装旧版本：
  ```bash
  pip install gradio==4.0.0
  ```

## 不使用 Web UI 的替代方案

如果不想安装 Gradio，可以使用：

### 1. 命令行工具
```bash
python style_shift.py -c photo.jpg --style-name anime -o output.jpg
```

### 2. Python API
```python
from style_shift import StyleTransfer
st = StyleTransfer()
result = st.transfer(content='photo.jpg', style_name='anime')
result.save('output.jpg')
```

---

*Web UI 是可选功能，核心功能无需 Gradio 即可使用*
