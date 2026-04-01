import gc
import logging
import numpy as np
import psutil
import torch

from PIL import Image
from pathlib import Path
from utils.config import cfg
from utils.dinov3 import DinoV3Model, get_all_crops
from utils.logging import custom_logging

DATA_DIR = Path(cfg['DATA_DIR'])

custom_logging()
logger = logging.getLogger(__name__)


def process_slice(model: DinoV3Model, em_slice: np.ndarray) -> np.ndarray:
    """Extract dense bilinear embeddings for a single 2-D EM slice.

    Returns a (H, W, 1024) float16 array.
    """
    inp_image = Image.fromarray(em_slice).convert('RGB')
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

    Output: <crop_path>/mito_embeddings.npz  shape (Z, H, W, 1024) float16
    Skips the crop if the output file already exists.
    """
    save_path = crop_path / 'mito_embeddings.npz'
    if save_path.exists():
        logger.info(f"  skipping {crop_path} — already done")
        return

    em = np.load(crop_path / 'em.npy', mmap_mode='r')
    slices = []

    with torch.no_grad():
        for z_idx in range(em.shape[0]):
            dense = process_slice(model, em[z_idx])
            slices.append(dense)
            logger.info(
                f"  [z={z_idx}/{em.shape[0]-1}] "
                f"RAM: {psutil.virtual_memory().used / 1e9:.1f}GB"
            )

    volume = np.stack(slices, axis=0)  # (Z, H, W, 1024)
    np.savez_compressed(save_path, embeddings=volume)
    logger.info(f"  saved {volume.shape} → {save_path}")

    del slices, volume
    gc.collect()
    torch.cuda.empty_cache()


def main() -> None:
    model = DinoV3Model()

    logger.info(
        f"[after model load] RAM: {psutil.virtual_memory().used / 1e9:.1f}GB"
        f" / {psutil.virtual_memory().total / 1e9:.1f}GB"
    )
    logger.info(
        f"[after model load] GPU: {torch.cuda.memory_allocated() / 1e9:.1f}GB"
        f" / {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB"
    )

    all_crops = get_all_crops(DATA_DIR)
    logger.info(f"Total crops: {len(all_crops)}")

    for dataset, crop, crop_path in all_crops:
        logger.info(f"{dataset}/{crop}")
        process_crop(model, crop_path)


if __name__ == "__main__":
    main()
