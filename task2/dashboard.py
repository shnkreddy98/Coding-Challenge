import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
from pathlib import Path
from sklearn.metrics import average_precision_score
from utils.config import cfg, model_tag
from utils.retrieval import (
    load_crop,
    get_mito_embedding,
    compute_similarity,
    build_similarity_map,
    get_best_z,
)

DATA_DIR = Path(cfg["DATA_DIR"])

st.set_page_config(layout="wide", page_title="Task 3 — Mito Retrieval")
st.markdown(
    """
    <style>
    .stApp {
        background-color: #000000;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

use_projected = st.sidebar.toggle(
    "Use projected embeddings (Task 4)",
    value=False,
    help="Uses 256-dim embeddings from trained linear probe. Run train_linear_probe.py + project_embeddings.py first.",
)

st.title(
    "Task 4 - Mitochondria Embedding Retrieval (Linear Probe)"
    if use_projected
    else "Task 3 - Mitochondria Embedding Retrieval"
)


def get_datasets() -> list[str]:
    return sorted([d.name for d in DATA_DIR.iterdir() if d.is_dir()])


def get_crops(dataset: str) -> list[str]:
    dataset_path = DATA_DIR / dataset
    tag = model_tag()
    return sorted(
        [
            c.name
            for c in dataset_path.iterdir()
            if c.is_dir() and (c / f"dense_embeddings_{tag}.npy").exists()
        ]
    )


@st.cache_resource(max_entries=6)
def cached_load_crop(dataset: str, crop: str, projected: bool = False) -> dict:
    return load_crop(dataset, crop, projected=projected)


def overlay_similarity(
    em_slice: np.ndarray,
    sim_map: np.ndarray,
    size: int = 400,
    mito_mask: np.ndarray | None = None,
) -> plt.Figure:
    """Render EM slice with cosine similarity heat map overlaid.
    Optionally draws GT mito mask contour in cyan."""
    dpi = 100
    fig, ax = plt.subplots(figsize=(size / dpi, size / dpi), dpi=dpi)
    ax.imshow(em_slice, cmap="gray", interpolation="nearest")
    vmin, vmax = float(sim_map.min()), float(sim_map.max())
    if vmax > vmin:
        ax.imshow(sim_map, cmap="hot", alpha=0.6, vmin=vmin, vmax=vmax)
    plt.colorbar(
        plt.cm.ScalarMappable(cmap="hot", norm=plt.Normalize(vmin, vmax)),
        ax=ax,
        fraction=0.046,
        label="cosine similarity",
    )
    if mito_mask is not None and mito_mask.any():
        ax.contour(mito_mask, levels=[0.5], colors="cyan", linewidths=1.5)
    ax.axis("off")
    plt.tight_layout(pad=0)
    return fig


def em_with_mito_overlay(em_slice: np.ndarray, mito_slice: np.ndarray) -> Image.Image:
    """Return RGB PIL image with mito mask pixels coloured red."""
    em_rgb = np.stack([em_slice] * 3, axis=-1).astype(np.uint8)
    em_rgb[mito_slice == 1] = [180, 80, 80]
    return Image.fromarray(em_rgb)


# --- Sidebar ---
st.sidebar.title("Dataset Selection")
st.sidebar.divider()
datasets = get_datasets()

st.sidebar.subheader("Query (source)")
query_dataset = st.sidebar.selectbox("Dataset", datasets, key="query_dataset")
crops = get_crops(query_dataset)
if not crops:
    st.warning(
        f"No dense embeddings found in {query_dataset}. Run extract_bilinear_embeddings.py first."
    )
    st.stop()
query_crop = st.sidebar.selectbox("Crop", crops, key="query_crop")

st.sidebar.divider()

st.sidebar.subheader("Intra-dataset")
st.sidebar.caption("Same dataset, different crop")
intra_crops = [c for c in get_crops(query_dataset) if c != query_crop]
if intra_crops:
    intra_crop = st.sidebar.selectbox("Crop", intra_crops, key="intra_crop")
    intra_dataset = query_dataset
else:
    st.sidebar.warning("No other crops in this dataset")
    intra_crop = None
    intra_dataset = None

st.sidebar.divider()

st.sidebar.subheader("Inter-dataset")
st.sidebar.caption("Must be a different dataset from query")
inter_datasets = [d for d in datasets if d != query_dataset]
if inter_datasets:
    inter_dataset = st.sidebar.selectbox("Dataset", inter_datasets, key="inter_dataset")
    inter_crop = st.sidebar.selectbox(
        "Crop", get_crops(inter_dataset), key="inter_crop"
    )
else:
    st.sidebar.warning("No other datasets available")
    inter_dataset = None
    inter_crop = None


# --- Data loading ---
query_data = cached_load_crop(query_dataset, query_crop, use_projected)
Z_QUERY = get_best_z(query_data)

intra_data = (
    cached_load_crop(intra_dataset, intra_crop, use_projected)
    if intra_dataset and intra_crop
    else None
)
Z_INTRA = get_best_z(intra_data) if intra_data else None

inter_data = (
    cached_load_crop(inter_dataset, inter_crop, use_projected)
    if inter_dataset and inter_crop
    else None
)
Z_INTER = get_best_z(inter_data) if inter_data else None


z = st.slider(
    "Query z slice", min_value=0, max_value=query_data["em"].shape[0] - 1, value=Z_QUERY
)

DISPLAY_SIZE = cfg["DISPLAY_SIZE"]

col1, col2, col3 = st.columns(3)

query_emb = get_mito_embedding(query_data, z)

with col1:
    st.subheader("Query")
    st.caption(f"{query_dataset}/{query_crop} | z={z}")

    em_slice = query_data["em"][z]
    mito_slice = query_data["mito_mask"][z]
    query_pil = em_with_mito_overlay(em_slice, mito_slice)
    query_pil_large = query_pil.resize((DISPLAY_SIZE, DISPLAY_SIZE), Image.NEAREST)
    st.image(query_pil_large)

    n_mito_px = int(mito_slice.sum())
    if query_emb is not None:
        st.caption(f"Query: mean of {n_mito_px} mito pixels (red)")
    else:
        st.warning("No mito pixels in this z-slice — adjust the slider")

with col2:
    st.subheader("Intra-dataset")
    if intra_data is not None:
        st.caption(f"{intra_dataset}/{intra_crop} | z={Z_INTRA}")
        if query_emb is not None:
            sim_intra = compute_similarity(query_emb, intra_data["embeddings"])
            sim_map_intra = build_similarity_map(sim_intra, Z_INTRA)
            fig = overlay_similarity(
                intra_data["em"][Z_INTRA],
                sim_map_intra,
                DISPLAY_SIZE,
                mito_mask=intra_data["mito_mask"][Z_INTRA],
            )
            st.pyplot(fig)
            plt.close()
            ap_intra = average_precision_score(
                intra_data["mito_mask"].flatten(), sim_intra.flatten()
            )
            pred_intra = (sim_intra > 0.5).astype(int)
            intersection = (pred_intra & intra_data["mito_mask"]).sum()
            union = (pred_intra | intra_data["mito_mask"]).sum()
            iou_intra = intersection / union if union > 0 else 0.0
            st.metric(
                "Average Precision",
                f"{ap_intra:.3f}",
                help="Cyan contour = GT mito mask",
            )
            st.metric("IoU (threshold=0.5)", f"{iou_intra:.3f}")
        else:
            st.warning("No mito pixels in query slice")
    else:
        st.warning("No intra-dataset crop available")

with col3:
    st.subheader("Inter-dataset")
    if inter_data is not None:
        st.caption(f"{inter_dataset}/{inter_crop} | z={Z_INTER}")
        if query_emb is not None:
            sim_inter = compute_similarity(query_emb, inter_data["embeddings"])
            sim_map_inter = build_similarity_map(sim_inter, Z_INTER)
            fig = overlay_similarity(
                inter_data["em"][Z_INTER],
                sim_map_inter,
                DISPLAY_SIZE,
                mito_mask=inter_data["mito_mask"][Z_INTER],
            )
            st.pyplot(fig)
            plt.close()
            ap_inter = average_precision_score(
                inter_data["mito_mask"].flatten(), sim_inter.flatten()
            )
            pred_inter = (sim_inter > 0.5).astype(int)
            intersection = (pred_inter & inter_data["mito_mask"]).sum()
            union = (pred_inter | inter_data["mito_mask"]).sum()
            iou_inter = intersection / union if union > 0 else 0.0
            st.metric(
                "Average Precision",
                f"{ap_inter:.3f}",
                help="Cyan contour = GT mito mask",
            )
            st.metric("IoU (threshold=0.5)", f"{iou_inter:.3f}")
        else:
            st.warning("No mito pixels in query slice")
    else:
        st.warning("No inter-dataset crop available")
