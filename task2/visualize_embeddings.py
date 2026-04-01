import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from PIL import Image
from sklearn.decomposition import PCA

DATA_DIR = Path("data")

st.set_page_config(layout="wide", page_title="Task 2 — Dense Embeddings")
st.markdown(
    """
    <style>
    .stApp { background-color: #000000; }
    </style>
    """,
    unsafe_allow_html=True
)
st.title("Task 2 — Dense Bilinear Embeddings")


def get_datasets() -> list[str]:
    return sorted([d.name for d in DATA_DIR.iterdir() if d.is_dir()])


def get_crops(dataset: str) -> list[str]:
    return sorted([
        c.name for c in (DATA_DIR / dataset).iterdir()
        if c.is_dir() and (c / 'mito_embeddings.npz').exists()
    ])


@st.cache_resource
def load_em_and_mask(dataset: str, crop: str):
    crop_path = DATA_DIR / dataset / crop
    em = np.load(crop_path / 'em.npy', mmap_mode='r')

    from skimage.transform import resize
    mito_mask_full = np.load(crop_path / 'mito_mask.npy', mmap_mode='r')
    mito_mask = (resize(
        mito_mask_full.astype(np.float32), em.shape,
        order=0, anti_aliasing=False
    ) > 0.5).astype(np.uint8)

    return em, mito_mask


@st.cache_data
def get_pca_rgb(dataset: str, crop: str, z: int) -> np.ndarray:
    """Load one z-slice of embeddings and reduce to RGB via PCA."""
    crop_path = DATA_DIR / dataset / crop
    data = np.load(crop_path / 'mito_embeddings.npz', mmap_mode='r')
    emb_slice = data['embeddings'][z].astype(np.float32)  # (H, W, 1024)
    H, W, D = emb_slice.shape

    pca = PCA(n_components=3)
    rgb = pca.fit_transform(emb_slice.reshape(H * W, D))  # (H*W, 3)

    for i in range(3):
        ch = rgb[:, i]
        rgb[:, i] = (ch - ch.min()) / (ch.max() - ch.min() + 1e-8) * 255

    return rgb.reshape(H, W, 3).astype(np.uint8)


# --- Sidebar ---
st.sidebar.title("Select Data")
datasets = get_datasets()
dataset = st.sidebar.selectbox("Dataset", datasets)
crops = get_crops(dataset)
if not crops:
    st.warning(f"No extracted embeddings found in {dataset}. Run extract_bilinear_embeddings.py first.")
    st.stop()
crop = st.sidebar.selectbox("Crop", crops)

em, mito_mask = load_em_and_mask(dataset, crop)

z = st.sidebar.slider("z slice", 0, em.shape[0] - 1, int(em.shape[0] // 2))

st.sidebar.caption(f"EM shape: {em.shape}")

# --- Main panels ---
col1, col2 = st.columns(2)

with col1:
    st.subheader("EM + Mito mask")
    em_rgb = np.stack([em[z]] * 3, axis=-1).astype(np.uint8)
    em_rgb[mito_mask[z] == 1] = [180, 80, 80]
    st.image(Image.fromarray(em_rgb))

with col2:
    st.subheader("Dense embeddings (PCA → RGB)")
    with st.spinner("Running PCA on slice..."):
        pca_img = get_pca_rgb(dataset, crop, z)
    st.image(Image.fromarray(pca_img))
