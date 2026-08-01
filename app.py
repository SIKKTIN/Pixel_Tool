"""
Perfect Pixel Tool — 本地 Gradio 界面

启动:  python app.py
打开:  浏览器访问 http://127.0.0.1:7860
"""

from __future__ import annotations

import numpy as np
import gradio as gr
from perfect_pixel import get_perfect_pixel

SAMPLE_METHODS = ["center", "median", "majority"]
SCALE_OPTIONS = [4, 8, 12, 16]


def _to_uint8_rgb(arr: np.ndarray) -> np.ndarray:
    """保证输入是 HxWx3 的 uint8 RGB 数组。"""
    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)
    if arr.shape[-1] == 4:  # RGBA -> RGB
        arr = arr[..., :3]
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    return arr


def _resize_nearest(img: np.ndarray, factor: int) -> np.ndarray:
    """用最近邻放大,纯 numpy 实现,避免再依赖 PIL 之外的 resize 库。"""
    h, w = img.shape[:2]
    out = np.repeat(np.repeat(img, factor, axis=0), factor, axis=1)
    return out


def refine(
    image: np.ndarray | None,
    sample_method: str,
    refine_intensity: float,
    fix_square: bool,
    preview_scale: int,
):
    if image is None:
        raise gr.Error("请先上传一张图片")

    rgb = _to_uint8_rgb(image)

    w, h, out = get_perfect_pixel(
        rgb,
        sample_method=sample_method,
        refine_intensity=refine_intensity,
        fix_square=fix_square,
        debug=False,
    )

    if w is None or h is None or out is None:
        raise gr.Error("未能从图片中检测到像素网格,试试更明显的像素风图。")

    if out.dtype != np.uint8:
        out = np.clip(out, 0, 255).astype(np.uint8)

    preview = _resize_nearest(out, preview_scale)

    info = (
        f"输出尺寸: **{w} × {h}**  "
        f"|  预览倍数: **{preview_scale}×**  "
        f"|  采样: `{sample_method}`"
    )
    return rgb, out, preview, info


with gr.Blocks(title="Perfect Pixel Tool") as demo:
    gr.Markdown(
        """
        # Perfect Pixel Tool
        上传一张像素风图片,自动检测网格并对齐到规整的网格上。
        """
    )

    with gr.Row():
        with gr.Column(scale=1):
            inp_img = gr.Image(
                label="输入图片",
                sources=["upload", "clipboard"],
                type="numpy",
                height=320,
            )
            sample_method = gr.Dropdown(
                SAMPLE_METHODS, value="center", label="采样方式"
            )
            refine_intensity = gr.Slider(
                minimum=0.0, maximum=0.5, step=0.05, value=0.3,
                label="网格对齐强度 refine_intensity",
            )
            fix_square = gr.Checkbox(value=True, label="近似正方形时强制输出正方形")
            preview_scale = gr.Radio(
                SCALE_OPTIONS, value=8, label="预览放大倍数"
            )
            run_btn = gr.Button("生成像素图", variant="primary")

        with gr.Column(scale=2):
            info_md = gr.Markdown("等待输入…")
            with gr.Tab("像素化结果"):
                out_native = gr.Image(label="像素化结果 (原始尺寸)", height=320)
            with gr.Tab("放大预览"):
                out_scaled = gr.Image(label=f"放大预览", height=480)
            with gr.Tab("原图"):
                inp_preview = gr.Image(label="原图", height=320)

    run_btn.click(
        refine,
        inputs=[inp_img, sample_method, refine_intensity, fix_square, preview_scale],
        outputs=[inp_preview, out_native, out_scaled, info_md],
    )

if __name__ == "__main__":
    demo.queue().launch(
        server_name="127.0.0.1",
        server_port=7860,
        show_error=True,
        theme=gr.themes.Soft(),
    )