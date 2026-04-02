import gc
import logging
import numpy as np
import torch

from PIL import Image
from pathlib import Path
from utils.config import cfg
from utils.dinov3 import DinoV3Model, get_all_crops
from utils.logging import custom_logging

DATA_DIR = Path(cfg["DATA_DIR"])

custom_logging()
logger = logging.getLogger(__name__)


def process_slice(model: DinoV3Model, em_slice: np.ndarray) -> np.ndarray:
    """Extract dense bilinear embeddings for a single 2-D EM slice.

    Returns a (H, W, 1024) float16 array.
    """
    inp_image = Image.fromarray(em_slice).convert("RGB")
    target_size = (inp_image.size[1], inp_image.size[0])

    embeddings = model.get_embeddings(inp_image)
    patch_tokens = model.get_patch_tokens(embeddings)
    dense = model.get_dense_embeddings(patch_tokens, target_size)  # (H, W, 1024)

    result = dense.detach().cpu().numpy().astype(np.float16)

    del embeddings, patch_tokens, dense
    torch.cuda.empty_cache()

    return result


def process_crop(model: DinoV3Model, crop_path: Path) -> None:
    """Extract and save dense embeddings for all z-slices of one crop.

    Output: <crop_path>/dense_embeddings.npy  shape (Z, H, W, 1024) float16
    Skips the crop if the output file already exists.
    """
    save_path = crop_path / "dense_embeddings.npy"
    if save_path.exists():
        logger.info(f"  skipping {crop_path} — already done")
        return

    em = np.load(crop_path / "em.npy", mmap_mode="r")
    Z, H, W = em.shape

    volume = np.lib.format.open_memmap(
        save_path, mode="w+", dtype=np.float16, shape=(Z, H, W, model.embed_dim)
    )

    for z_idx in range(Z):
        volume[z_idx] = process_slice(model, em[z_idx])
        logger.info(f"  [z={z_idx}/{Z - 1}]")

    logger.info(f"  saved {volume.shape} → {save_path}")

    del volume
    gc.collect()
    torch.cuda.empty_cache()


def extract_bilinear() -> None:
    model = DinoV3Model()

    all_crops = get_all_crops(DATA_DIR)
    logger.info(f"Total crops: {len(all_crops)}")

    for dataset, crop, crop_path in all_crops:
        logger.info(f"{dataset}/{crop}")
        process_crop(model, crop_path)


if __name__ == "__main__":
    extract_bilinear()
