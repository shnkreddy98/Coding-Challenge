import numpy as np

from pathlib import Path
from skimage.transform import resize
from sklearn.metrics.pairwise import cosine_similarity
from utils.config import cfg

DATA_DIR = Path(cfg["DATA_DIR"])


def load_mito_mask(crop_path: Path, em_shape: tuple) -> np.ndarray:
    mito_mask_full = np.load(crop_path / "mito_mask.npy", mmap_mode="r")
    return (
        resize(
            mito_mask_full.astype(np.float32), em_shape, order=0, anti_aliasing=False
        )
        > 0.5
    ).astype(np.uint8)


def load_crop(dataset: str, crop: str, projected: bool = False) -> dict:
    crop_path = DATA_DIR / dataset / crop
    proj_path = crop_path / "mito_embeddings.npz"

    if projected and proj_path.exists():
        data = np.load(proj_path)
        embeddings = data["embeddings"]  # (Z, H, W, 256) float16
    else:
        embeddings = np.load(
            crop_path / "dense_embeddings.npy", mmap_mode="r"
        )  # (Z, H, W, 1024) float16

    em = np.load(crop_path / "em.npy")
    mito_mask_ds = load_mito_mask(crop_path, em.shape)

    return {
        "embeddings": embeddings,
        "em": em,
        "mito_mask": mito_mask_ds,
        "dataset": dataset,
        "crop": crop,
    }


def get_mito_embedding(crop_data: dict, z: int) -> np.ndarray | None:
    """Mean embedding of all mito pixels in a z-slice. Returns None if no mito pixels."""
    mito_slice = crop_data["mito_mask"][z]  # (H, W)
    emb_slice = crop_data["embeddings"][z]  # (H, W, 1024)
    mito_pixels = emb_slice[mito_slice == 1].astype(np.float32)  # (N, 1024)
    if len(mito_pixels) == 0:
        return None
    return mito_pixels.mean(axis=0)  # (1024,)


def compute_similarity(query_emb: np.ndarray, target_emb: np.ndarray) -> np.ndarray:
    """
    Cosine similarity between one query vector and the full target volume.
    Computed slice-by-slice to avoid materialising the full volume as float32.
    target_emb: (Z, H, W, D)
    returns:    (Z, H, W)
    """
    Z, H, W, D = target_emb.shape
    result = np.empty((Z, H, W), dtype=np.float32)
    q = query_emb.reshape(1, -1)
    for z in range(Z):
        flat = target_emb[z].reshape(H * W, D).astype(np.float32)
        result[z] = cosine_similarity(q, flat)[0].reshape(H, W)
    return result


def build_similarity_map(sim_scores: np.ndarray, z: int) -> np.ndarray:
    """Return the (H, W) similarity slice for a given z."""
    return sim_scores[z].astype(np.float32)


def get_best_z(crop_data: dict) -> int:
    """Return z slice with most mito pixels."""
    return int(crop_data["mito_mask"].sum(axis=(1, 2)).argmax())
