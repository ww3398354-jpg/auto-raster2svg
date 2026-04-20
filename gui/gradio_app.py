"""Gradio web GUI for auto-raster2svg."""
from __future__ import annotations

import logging
import os
import tempfile
from pathlib import Path

import gradio as gr
import numpy as np
from PIL import Image

from src.pipeline import run

logger = logging.getLogger(__name__)


def _process(
    image: np.ndarray | None,
    remove_bg: bool,
    remove_watermark: bool,
    industrial: bool,
    stroke_mode: bool,
) -> tuple:
    """Process the uploaded image and return (preview_image, svg_path)."""
    if image is None:
        return None, None

    tmp_dir = tempfile.mkdtemp()
    in_path = os.path.join(tmp_dir, "input.png")
    out_path = os.path.join(tmp_dir, "output.svg")

    # Save uploaded numpy array as PNG
    Image.fromarray(image.astype("uint8")).save(in_path)

    try:
        run(
            in_path,
            out_path,
            stroke_mode=stroke_mode,
            remove_bg=remove_bg,
            remove_watermark=remove_watermark,
            industrial=industrial,
        )
    except Exception as exc:  # noqa: BLE001
        logger.error("Pipeline error: %s", exc)
        return None, None

    # Render SVG back to PNG for preview
    try:
        import cairosvg  # noqa: PLC0415
        import io  # noqa: PLC0415

        h, w = image.shape[:2]
        png_bytes = cairosvg.svg2png(url=out_path, output_width=w, output_height=h)
        preview = np.array(Image.open(io.BytesIO(png_bytes)).convert("RGB"))
    except Exception:  # noqa: BLE001
        preview = image  # fall back to original

    return preview, out_path


def build_app() -> gr.Blocks:
    with gr.Blocks(title="auto-raster2svg") as demo:
        gr.Markdown(
            "# 🎨 auto-raster2svg\n"
            "Upload any raster image and download a layered, editable SVG."
        )

        with gr.Row():
            with gr.Column():
                img_input = gr.Image(label="Upload image", type="numpy")
                with gr.Row():
                    cb_bg = gr.Checkbox(label="Remove background", value=False)
                    cb_wm = gr.Checkbox(label="Remove watermark", value=False)
                with gr.Row():
                    cb_ind = gr.Checkbox(label="Industrial (quality retry)", value=False)
                    cb_stroke = gr.Checkbox(label="Stroke only", value=False)
                btn = gr.Button("Vectorize ▶", variant="primary")
            with gr.Column():
                img_preview = gr.Image(label="SVG preview")
                file_out = gr.File(label="Download SVG")

        btn.click(
            fn=_process,
            inputs=[img_input, cb_bg, cb_wm, cb_ind, cb_stroke],
            outputs=[img_preview, file_out],
        )

    return demo


if __name__ == "__main__":
    app = build_app()
    app.launch()
