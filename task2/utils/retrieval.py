import numpy as np

from pathlib import Path
from skimage.transform import resize
from sklearn.metrics.pairwise import cosine_similarity
from utils.config import cfg

DATA_DIR = Path(cfg['DATA_DIR'])


def load_mito_mask(crop_path: Path, em_shape: tuple) -> np.ndarray:
    mito_mask_full = np.load(crop_path / 'mito_mask.npy', mmap_mode='r')
    return (resize(
        mito_mask_full.astype(np.float32),
        em_shape,
        order=0,
        anti_aliasing=False
    ) > 0.5).astype(np.uint8)


def load_crop(dataset: str, crop: str) -> dict:
    crop_path = DATA_DIR / dataset / crop
    data = np.load(crop_path / 'mito_embeddings_projected.npz')
    em = np.load(crop_path / 'em.npy')
    mito_mask_ds = load_mito_mask(crop_path, em.shape)

    return {
        'embeddings': data['embeddings'],   # (Z, H, W, 256) float16
        'em':         em,
        'mito_mask':  mito_mask_ds,
        'dataset':    dataset,
        'crop':       crop,
    }


def get_pixel_embedding(crop_data: dict, z: int, y: int, x: int) -> np.ndarray:
    """Direct index — every pixel has an embedding."""
    return crop_data['embeddings'][z, y, x].astype(np.float32)  # (256,)


def compute_similarity(query_emb: np.ndarray, target_emb: np.ndarray) -> np.ndarray:
    """
    Cosine similarity between one query vector and the full target volume.
    target_emb: (Z, H, W, D)
    returns:    (Z, H, W)
    """
    Z, H, W, D = target_emb.shape
    flat = target_emb.reshape(Z * H * W, D).astype(np.float32)
    sims = cosine_similarity(query_emb.reshape(1, -1), flat)[0]  # (Z*H*W,)
    return sims.reshape(Z, H, W)


def build_similarity_map(sim_scores: np.ndarray, z: int) -> np.ndarray:
    """Return the (H, W) similarity slice for a given z."""
    return sim_scores[z].astype(np.float32)


def get_best_z(crop_data: dict) -> int:
    """Return z slice with most mito pixels."""
    return int(crop_data['mito_mask'].sum(axis=(1, 2)).argmax())
