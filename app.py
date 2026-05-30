"""LanPaint · Streamlit inpainting app."""

from __future__ import annotations

import io
import os
import time
from pathlib import Path
from typing import Optional

import tomllib
import numpy as np
import streamlit as st
import torch
from PIL import Image
from streamlit_drawable_canvas import st_canvas

from lanpaint_pipeline import LanPaintConfig, LanPaintInpaintPipeline
from lanpaint_pipeline.registry import create_adapter, get_model_spec, list_models

# ── Config ────────────────────────────────────────────────────────────────────

CANVAS_MAX_W = 720

_CONFIG_PATH = Path(__file__).parent / "config.toml"


@st.cache_data
def _load_config() -> dict:
    if _CONFIG_PATH.exists():
        with open(_CONFIG_PATH, "rb") as f:
            return tomllib.load(f)
    return {}


def _model_cfg(key: str) -> dict:
    cfg = _load_config()
    return cfg.get("models", {}).get(key, {})


_MODEL_LABELS: dict[str, str] = {
    "flux-klein": "Flux.2 Klein 9B",
    "sd3": "Stable Diffusion 3",
    "z-image": "Z-Image Turbo",
    "qwen": "Qwen Image Edit",
}

# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="LanPaint · Inpainting",
    page_icon="🖌️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
section[data-testid="stSidebar"] { background: #161b22; }
section[data-testid="stSidebar"] [data-testid="stMarkdown"] h2,
section[data-testid="stSidebar"] [data-testid="stMarkdown"] h3 {
    color: #e6edf3;
}
.stButton > button { border-radius: 8px; font-weight: 600; }
.stDownloadButton > button { border-radius: 8px; font-weight: 600; }
.canvas-hint {
    font-size: 0.82rem;
    color: #8b949e;
    margin-top: 0.25rem;
}
.result-header {
    font-size: 0.95rem;
    color: #8b949e;
    border-top: 1px solid #30363d;
    padding-top: 0.75rem;
    margin-top: 0.5rem;
}
</style>
""",
    unsafe_allow_html=True,
)

# ── Session state init ────────────────────────────────────────────────────────

_defaults = {
    "adapter": None,
    "loaded_key": None,
    "canvas_key": 0,
    "result_img": None,
    "result_elapsed": 0.0,
    "error_msg": None,
}
for k, v in _defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ── Model loading (cached) ────────────────────────────────────────────────────


@st.cache_resource(show_spinner=False)
def _load_adapter(model_key: str, model_id: str, local_files_only: bool):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    kwargs: dict = {}
    if local_files_only:
        kwargs["local_files_only"] = True
    eff_id = model_id.strip() or None
    return create_adapter(model_key, device=device, model_id=eff_id, **kwargs)


# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## ⚙️ Settings")

    # ── Model selection ──────────────────────────────────────────────────────
    st.markdown("### Model")
    all_keys = list_models()
    model_key: str = st.selectbox(
        "model",
        all_keys,
        index=0,
        format_func=lambda k: _MODEL_LABELS.get(k, k),
        label_visibility="collapsed",
    )
    spec = get_model_spec(model_key)
    mcfg = _model_cfg(model_key)
    default_path = mcfg.get("path", "") or spec.default_model_id
    default_local = mcfg.get("local_files_only", False)

    model_id: str = st.text_input(
        "Model path / Hub ID",
        value=default_path,
        help="Local filesystem path or HuggingFace Hub ID — edit config.toml to change the default",
    )
    local_files_only: bool = st.checkbox(
        "Local files only",
        value=default_local,
        help="Skip Hub network requests",
    )

    status_slot = st.empty()
    if st.button("🚀 Load model", type="primary", use_container_width=True):
        with st.spinner(f"Loading {_MODEL_LABELS.get(model_key, model_key)}…"):
            try:
                adapter = _load_adapter(model_key, model_id, local_files_only)
                st.session_state.adapter = adapter
                st.session_state.loaded_key = model_key
                st.session_state.result_img = None
                st.session_state.error_msg = None
            except Exception as exc:
                st.session_state.error_msg = str(exc)

        if st.session_state.error_msg:
            status_slot.error(f"Load failed: {st.session_state.error_msg}")
        elif st.session_state.loaded_key == model_key:
            status_slot.success(f"✓ {_MODEL_LABELS.get(model_key, model_key)} ready")

    if (
        st.session_state.adapter is not None
        and st.session_state.loaded_key == model_key
        and not st.session_state.error_msg
    ):
        status_slot.success(f"✓ {_MODEL_LABELS.get(model_key, model_key)} ready")

    st.divider()

    # ── Generation params ────────────────────────────────────────────────────
    st.markdown("### Generation")
    guidance_scale: float = st.slider(
        "Guidance scale",
        1.0,
        20.0,
        float(spec.default_params.get("guidance_scale", 5.0)),
        step=0.5,
    )
    num_steps: int = st.slider(
        "Inference steps",
        5,
        100,
        int(spec.default_params.get("num_inference_steps", 20)),
    )
    seed: int = st.number_input(
        "Seed", min_value=0, max_value=2**31 - 1, value=42
    )

    with st.expander("Advanced · LanPaint"):
        lp_n_steps: int = st.slider("Langevin steps / scheduler step", 1, 10, 2)
        lp_friction: float = st.slider("Friction", 1.0, 50.0, 15.0)
        lp_lambda: float = st.slider("Lambda (λ)", 1.0, 32.0, 8.0)
        lp_beta: float = st.slider("Beta (β)", 0.1, 5.0, 1.0, step=0.1)
        lp_step_size: float = st.slider("Step size", 0.01, 1.0, 0.2, step=0.01)
        lp_early_stop: int = st.slider("Early-stop steps", 0, 5, 1)
        lp_blend_overlap: int = st.slider("Blend overlap (px)", 0, 20, 9)

# ── Main area ─────────────────────────────────────────────────────────────────

st.markdown("# 🖌️ LanPaint")
st.markdown("Upload an image · paint the region to edit · describe the change · generate.")

# Track these across columns (set in col_left, read in col_right)
orig_img: Optional[Image.Image] = None
mask_pil: Optional[Image.Image] = None
ref_img: Optional[Image.Image] = None

col_left, col_right = st.columns([1.15, 1], gap="large")

# ── Left: image uploader + mask canvas ───────────────────────────────────────

with col_left:
    st.markdown("#### Input image")
    uploaded = st.file_uploader(
        "upload",
        type=["png", "jpg", "jpeg", "webp"],
        label_visibility="collapsed",
    )

    if uploaded is not None:
        orig_img = Image.open(uploaded).convert("RGB")
        ow, oh = orig_img.size

        # Scale image to fit canvas display width
        if ow > CANVAS_MAX_W:
            scale = CANVAS_MAX_W / ow
            cw, ch = CANVAS_MAX_W, int(oh * scale)
        else:
            cw, ch = ow, oh

        canvas_img = orig_img.resize((cw, ch), Image.BICUBIC)

        st.markdown("#### Mask — paint the area to edit")
        ctrl1, ctrl2, ctrl3 = st.columns([3, 1.2, 1.2])
        with ctrl1:
            brush_size: int = st.slider(
                "Brush size", 5, 120, 30, label_visibility="collapsed"
            )
        with ctrl2:
            st.metric("Resolution", f"{ow}×{oh}", label_visibility="visible")
        with ctrl3:
            if st.button("🗑 Clear mask", use_container_width=True):
                st.session_state.canvas_key += 1

        canvas_result = st_canvas(
            fill_color="rgba(255, 75, 75, 0.35)",
            stroke_width=brush_size,
            stroke_color="#FF4B4B",
            background_image=canvas_img,
            update_streamlit=True,
            height=ch,
            width=cw,
            drawing_mode="freedraw",
            key=f"canvas_{st.session_state.canvas_key}",
            display_toolbar=False,
        )

        st.markdown(
            '<p class="canvas-hint">🔴 Paint over areas to <strong>edit</strong> '
            "— unpainted regions will be <strong>preserved</strong>.</p>",
            unsafe_allow_html=True,
        )

        # Build keep-mask from drawn alpha layer
        if canvas_result.image_data is not None:
            alpha = canvas_result.image_data[:, :, 3]
            if alpha.max() > 0:
                # drawn (alpha > 0) → 0 (edit), blank → 255 (keep)
                mask_np = np.where(alpha > 0, 0, 255).astype(np.uint8)
                mask_canvas = Image.fromarray(mask_np, mode="L")
                mask_pil = mask_canvas.resize((ow, oh), Image.NEAREST)

        # ── Reference image (Flux Klein only) ───────────────────────────────
        with st.expander("📎 Reference image — optional, Flux only"):
            st.caption(
                "Upload a second image as visual inspiration. "
                "The model will use it alongside your prompt to guide generation "
                "in the masked region."
            )
            ref_uploaded = st.file_uploader(
                "Reference image",
                type=["png", "jpg", "jpeg", "webp"],
                key="ref_upload",
                label_visibility="collapsed",
            )
            if ref_uploaded is not None:
                ref_img = Image.open(ref_uploaded).convert("RGB")
                st.image(ref_img, use_container_width=True)

    else:
        st.markdown(
            """
<div style="
    border: 2px dashed #30363d;
    border-radius: 12px;
    padding: 3.5rem 2rem;
    text-align: center;
    color: #8b949e;
    margin-top: 0.5rem;
">
    <div style="font-size: 2.5rem;">🖼️</div>
    <div style="font-size: 1rem; margin-top: 0.5rem;">
        Upload an image above to start painting
    </div>
</div>
""",
            unsafe_allow_html=True,
        )

# ── Right: prompt + generate + result ────────────────────────────────────────

with col_right:
    st.markdown("#### Prompt")
    prompt: str = st.text_area(
        "prompt",
        placeholder=(
            "Describe the edit, e.g. 'change the sky to a dramatic sunset'"
            " or 'add a red dress'"
        ),
        height=110,
        label_visibility="collapsed",
    )
    negative_prompt: str = st.text_input(
        "Negative prompt",
        placeholder="Artifacts, blur, distortion…",
    )

    # Pre-flight checks
    issues: list[str] = []
    if st.session_state.adapter is None:
        issues.append("⬅ Load a model from the sidebar first.")
    elif st.session_state.loaded_key != model_key:
        issues.append(f"⬅ Model selection changed — reload the model.")
    if orig_img is None:
        issues.append("⬆ Upload an image.")
    if orig_img is not None and mask_pil is None:
        issues.append("🖌 Paint the area you want to edit on the canvas.")
    if not prompt.strip():
        issues.append("✏ Enter a prompt describing the desired edit.")

    can_generate = len(issues) == 0

    for msg in issues:
        st.info(msg)

    generate_btn = st.button(
        "✨ Generate",
        type="primary",
        use_container_width=True,
        disabled=not can_generate,
    )

    # ── Run generation ───────────────────────────────────────────────────────
    if generate_btn and can_generate:
        lp_config = LanPaintConfig(
            n_steps=lp_n_steps,
            friction=lp_friction,
            chara_lambda=lp_lambda,
            beta=lp_beta,
            step_size=lp_step_size,
            early_stop=lp_early_stop,
            blend_overlap=lp_blend_overlap,
        )
        pipeline = LanPaintInpaintPipeline(
            st.session_state.adapter, config=lp_config
        )

        with st.spinner("Generating… this may take a minute."):
            try:
                t0 = time.time()
                result = pipeline(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    image=orig_img,
                    mask_image=mask_pil,
                    ref_images=[ref_img] if ref_img is not None else None,
                    guidance_scale=guidance_scale,
                    num_inference_steps=num_steps,
                    seed=int(seed),
                )
                elapsed = time.time() - t0
                st.session_state.result_img = result.images[0]
                st.session_state.result_elapsed = elapsed
                st.session_state.error_msg = None
            except Exception as exc:
                st.session_state.error_msg = str(exc)

        if st.session_state.error_msg:
            st.error(f"Generation failed: {st.session_state.error_msg}")

    # ── Display result ───────────────────────────────────────────────────────
    if st.session_state.result_img is not None:
        st.markdown(
            f'<p class="result-header">Result &nbsp;·&nbsp; '
            f"{st.session_state.result_elapsed:.1f}s</p>",
            unsafe_allow_html=True,
        )
        st.image(st.session_state.result_img, use_container_width=True)

        buf = io.BytesIO()
        st.session_state.result_img.save(buf, format="PNG")
        st.download_button(
            "⬇️ Download PNG",
            data=buf.getvalue(),
            file_name="lanpaint_output.png",
            mime="image/png",
            use_container_width=True,
        )
