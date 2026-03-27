#!/usr/bin/env python3
"""Gradio Web UI for StyleShift - 风转神经风格迁移"""

import gradio as gr
from PIL import Image
import tempfile
from pathlib import Path

from style_shift.core.style_transfer import StyleTransfer, StyleTransferConfig


BUILTIN_STYLES = {
    "动漫": "anime",
    "梵高": "vangogh",
    "莫奈": "monet",
    "浮世绘": "ukiyoe",
    "马赛克": "mosaic",
    "素描": "sketch",
    "水彩": "watercolor",
}


def style_transfer(
    content_image,
    style_type,
    custom_style_image,
    style_strength,
    max_size,
    preserve_color
):
    """Gradio 风格迁移封装函数。"""
    if content_image is None:
        return None
    
    # 初始化 StyleTransfer
    config = StyleTransferConfig(
        alpha=style_strength,
        max_size=max_size,
        preserve_color=preserve_color,
        device='cpu'  # Web UI 使用 CPU
    )
    
    st = StyleTransfer(config)
    
    # 确定风格来源
    style = None
    style_name = None
    
    if style_type == "自定义" and custom_style_image is not None:
        style = custom_style_image
    elif style_type != "自定义":
        style_name = BUILTIN_STYLES.get(style_type)
    
    if style is None and style_name is None:
        return None
    
    # 执行风格迁移
    result = st.transfer(
        content=content_image,
        style=style,
        style_name=style_name
    )
    
    return result


def create_ui():
    """创建 Gradio 界面。"""
    with gr.Blocks(
        title="StyleShift - 神经风格迁移",
    ) as demo:
        gr.Markdown("""
        # 🎨 StyleShift - 风转
        
        上传内容图像并选择风格，使用 AdaIN 进行神经风格迁移。
        
        **功能特点：**
        - 7 种内置艺术风格
        - 支持自定义风格上传
        - 可调节风格强度
        - 颜色保留选项
        """)
        
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### 输入")
                
                content_input = gr.Image(
                    label="内容图像",
                    type="pil",
                    height=300
                )
                
                with gr.Row():
                    style_type = gr.Dropdown(
                        choices=list(BUILTIN_STYLES.keys()) + ["自定义"],
                        value="动漫",
                        label="风格类型",
                        interactive=True
                    )
                    custom_style = gr.Image(
                        label="自定义风格",
                        type="pil",
                        height=150,
                        visible=False
                    )
                
                style_strength = gr.Slider(
                    0.0, 1.0,
                    value=1.0,
                    step=0.1,
                    label="风格强度 (Alpha)"
                )
                
                max_size = gr.Slider(
                    256, 1024,
                    value=512,
                    step=128,
                    label="最大输出尺寸"
                )
                
                preserve_color = gr.Checkbox(
                    label="保留原始颜色",
                    value=False
                )
                
                submit_btn = gr.Button(
                    "🎨 应用风格迁移",
                    variant="primary",
                    size="lg"
                )
            
            with gr.Column(scale=1):
                gr.Markdown("### 输出结果")
                
                output_image = gr.Image(
                    label="风格化结果",
                    type="pil",
                    height=400
                )
                
                download_btn = gr.File(
                    label="下载结果",
                    visible=False
                )
        
        # 示例图像
        gr.Markdown("### 示例")
        gr.Examples(
            examples=[
                ["https://picsum.photos/seed/photo1/512/512.jpg", "动漫"],
                ["https://picsum.photos/seed/photo2/512/512.jpg", "梵高"],
                ["https://picsum.photos/seed/photo3/512/512.jpg", "莫奈"],
            ],
            inputs=[content_input, style_type],
        )
        
        # 切换自定义风格可见性
        style_type.change(
            fn=lambda x: gr.update(visible=(x == "自定义")),
            inputs=style_type,
            outputs=custom_style
        )
        
        # 提交按钮
        submit_btn.click(
            fn=style_transfer,
            inputs=[
                content_input,
                style_type,
                custom_style,
                style_strength,
                max_size,
                preserve_color
            ],
            outputs=output_image
        )
    
    return demo


if __name__ == "__main__":
    demo = create_ui()
    print("正在启动 StyleShift Web UI...")
    print("请在浏览器中打开 http://localhost:7860")
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False
    )
