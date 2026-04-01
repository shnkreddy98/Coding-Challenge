import gc
import logging
import numpy as np
import psutil
import torch
import torch.nn as nn
import torch.nn.functional as F

from PIL import Image
from pathlib import Path
from utils.config import cfg
from utils.dinov3 import DinoV3Model, get_all_crops
from utils.retrieval import load_mito_mask
from utils.logging import custom_logging

DATA_DIR = Path(cfg['DATA_DIR'])
N_EPOCHS = cfg['N_EPOCHS']
LR = cfg['LR']
EMBED_DIM = cfg['EMBED_DIM']
PROJ_DIM = cfg['PROJ_DIM']

custom_logging()
logger = logging.getLogger(__name__)


def train_slice(
    model: DinoV3Model,
    projection: nn.Linear,
    seg_head: nn.Linear,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    em_slice: np.ndarray,
    mito_slice: np.ndarray,
) -> float:
    """Forward + backward pass for one z-slice. Returns the scalar loss."""
    inp_image = Image.fromarray(em_slice).convert('RGB')
    target_size = (inp_image.size[1], inp_image.size[0])

    embeddings = model.get_embeddings(inp_image)
    patch_tokens = model.get_patch_tokens(embeddings)
    dense = model.get_dense_embeddings(patch_tokens, target_size)

    projected = projection(dense)       # (H, W, 256)
    pred = seg_head(projected)          # (H, W, 1)

    target = torch.tensor(
        mito_slice.astype(np.float32)
    ).unsqueeze(-1).to(model.device)

    loss = criterion(pred, target)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    loss_val = loss.item()
    del embeddings, patch_tokens, dense, projected, pred, target
    torch.cuda.empty_cache()

    return loss_val


def train_epoch(
    model: DinoV3Model,
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

        for z_idx in range(em.shape[0]):
            if mito_mask_ds[z_idx].sum() == 0:
                continue
            loss_val = train_slice(
                model, projection, seg_head, optimizer, criterion,
                em[z_idx], mito_mask_ds[z_idx],
            )
            total_loss += loss_val
            total_slices += 1

        del mito_mask_ds
        gc.collect()

    avg_loss = total_loss / total_slices if total_slices > 0 else 0.0
    logger.info(f"  epoch {epoch+1}/{N_EPOCHS} avg_loss={avg_loss:.4f}")
    return avg_loss


def train(model: DinoV3Model, all_crops: list) -> nn.Linear:
    """Train projection + segmentation head; return the trained projection."""
    projection = nn.Linear(EMBED_DIM, PROJ_DIM).to(model.device)
    seg_head = nn.Linear(PROJ_DIM, 1).to(model.device)

    optimizer = torch.optim.Adam(
        list(projection.parameters()) + list(seg_head.parameters()),
        lr=LR,
    )
    criterion = nn.BCEWithLogitsLoss()

    projection.train()
    seg_head.train()

    for epoch in range(N_EPOCHS):
        train_epoch(model, projection, seg_head, optimizer, criterion, all_crops, epoch)

    return projection


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

    projection = train(model, all_crops)

    torch.save(projection.state_dict(), DATA_DIR / 'projection.pt')
    logger.info(f"projection weights saved → {DATA_DIR / 'projection.pt'}")


if __name__ == "__main__":
    main()
