import argparse
import logging
import numpy as np
import torch
import torch.nn as nn

from pathlib import Path
from utils.config import cfg
from utils.dinov3 import get_all_crops
from utils.logging import custom_logging

DATA_DIR = Path(cfg['DATA_DIR'])
EMBED_DIM = cfg['EMBED_DIM']
PROJ_DIM = cfg['PROJ_DIM']

custom_logging()
logger = logging.getLogger(__name__)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Project bilinear embeddings using a trained linear probe")
    parser.add_argument('-p', '--projection', required=True, help="Path to projection.pt")
    args = parser.parse_args()

    proj_path = Path(args.projection)
    if not proj_path.exists():
        logger.error(f"projection.pt not found at {proj_path}")
        exit(1)

    projection = nn.Linear(EMBED_DIM, PROJ_DIM)
    projection.load_state_dict(torch.load(proj_path, map_location='cpu'))
    projection.eval()
    logger.info(f"projection loaded from {proj_path}")

    all_crops = get_all_crops(DATA_DIR)

    for dataset, crop, crop_path in all_crops:
        src = crop_path / 'mito_embeddings.npz'
        dst = crop_path / 'mito_embeddings_projected.npz'

        if not src.exists():
            logger.info(f"  skipping {dataset}/{crop} — mito_embeddings.npz not found")
            continue
        if dst.exists():
            logger.info(f"  skipping {dataset}/{crop} — already done")
            continue

        logger.info(f"{dataset}/{crop}")
        data = np.load(src, mmap_mode='r')
        embeddings = data['embeddings']  # (Z, H, W, 1024) float16
        Z, H, W, D = embeddings.shape

        slices = []
        with torch.no_grad():
            for z_idx in range(Z):
                flat = torch.tensor(embeddings[z_idx].astype(np.float32).reshape(H * W, D))
                out = projection(flat).numpy().astype(np.float16)  # (H*W, 256)
                slices.append(out.reshape(H, W, PROJ_DIM))
                logger.info(f"  [z={z_idx}/{Z-1}]")

        volume = np.stack(slices, axis=0)  # (Z, H, W, 256)
        np.savez_compressed(dst, embeddings=volume)
        logger.info(f"  saved {volume.shape} → {dst}")
        del slices, volume
