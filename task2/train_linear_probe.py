import gc
import logging
import numpy as np
import torch
import torch.nn as nn

from pathlib import Path
from transformers import AutoConfig
from utils.config import cfg
from utils.dinov3 import get_all_crops
from utils.retrieval import load_mito_mask
from utils.logging import custom_logging

DATA_DIR = Path(cfg['DATA_DIR'])
N_EPOCHS = cfg['N_EPOCHS']
LR = cfg['LR']
PROJ_DIM = cfg['PROJ_DIM']
EMBED_DIM = AutoConfig.from_pretrained(cfg['MODEL_NAME']).hidden_size

custom_logging()
logger = logging.getLogger(__name__)


def train_slice(
    device: torch.device,
    projection: nn.Linear,
    seg_head: nn.Linear,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    mito_slice: np.ndarray,
    dense: np.ndarray
) -> float:
    """Forward + backward pass for one z-slice. Returns the scalar loss."""

    dense_t = torch.tensor(dense.astype(np.float32)).to(device)

    projected = projection(dense_t)       # (H, W, 256)
    pred = seg_head(projected)          # (H, W, 1)

    target = torch.tensor(
        mito_slice.astype(np.float32)
    ).unsqueeze(-1).to(device)

    loss = criterion(pred, target)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    loss_val = loss.item()
    del dense_t, projected, pred, target
    torch.cuda.empty_cache()

    return loss_val


def train_epoch(
    device: torch.device,
    projection: nn.Linear,
    seg_head: nn.Linear,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    all_crops: list,
    epoch: int,
) -> float:
    """Run one full epoch over all crops and slices. Returns average loss."""
    total_loss = 0.0
    total_slices = 0

    for dataset, crop, crop_path in all_crops:
        logger.info(f"  epoch {epoch+1} | {dataset}/{crop}")

        em = np.load(crop_path / 'em.npy', mmap_mode='r')
        mito_mask_ds = load_mito_mask(crop_path, em.shape)
        dense_embeddings = np.load(crop_path / 'dense_embeddings.npy', mmap_mode='r')  # (Z, H, W, 1024)

        for z_idx in range(em.shape[0]):
            if mito_mask_ds[z_idx].sum() == 0:
                continue
            loss_val = train_slice(
                device,
                projection, seg_head, optimizer, criterion,
                mito_mask_ds[z_idx],
                dense_embeddings[z_idx]
            )
            total_loss += loss_val
            total_slices += 1

        del mito_mask_ds
        gc.collect()

    avg_loss = total_loss / total_slices if total_slices > 0 else 0.0
    logger.info(f"  epoch {epoch+1}/{N_EPOCHS} avg_loss={avg_loss:.4f}")
    return avg_loss


def train(all_crops: list) -> nn.Linear:
    """Train projection + segmentation head; return the trained projection."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    projection = nn.Linear(EMBED_DIM, PROJ_DIM).to(device)
    seg_head = nn.Linear(PROJ_DIM, 1).to(device)

    optimizer = torch.optim.Adam(
        list(projection.parameters()) + list(seg_head.parameters()),
        lr=LR,
    )
    criterion = nn.BCEWithLogitsLoss()

    projection.train()
    seg_head.train()

    for epoch in range(N_EPOCHS):
        train_epoch(device, projection, seg_head, optimizer, criterion, all_crops, epoch)

    return projection


def main() -> None:
    all_crops = get_all_crops(DATA_DIR)
    logger.info(f"Total crops: {len(all_crops)}")

    projection = train(all_crops)

    torch.save(projection.state_dict(), DATA_DIR / 'projection.pt')
    logger.info(f"projection weights saved → {DATA_DIR / 'projection.pt'}")


if __name__ == "__main__":
    main()
