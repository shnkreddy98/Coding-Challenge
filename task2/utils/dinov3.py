import torch
import torch.nn.functional as F

from pathlib import Path
from PIL import Image
from transformers import AutoImageProcessor, AutoModel
from utils.config import cfg

MODEL_NAME = cfg['MODEL_NAME']
PROJ_DIM = cfg['PROJ_DIM']


class DinoV3Model():
    def __init__(self, model_name: str = MODEL_NAME):
        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name, device_map="auto")
        self.device = next(self.model.parameters()).device
        self.embed_dim = self.model.config.hidden_size

    def get_embeddings(self, image: Image):
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        with torch.inference_mode():
            outputs = self.model(**inputs)
        return outputs

    def get_patch_tokens(self, embeddings):
        # last_hidden_state layout: [CLS, reg_0, ..., reg_N, patch_0, ..., patch_M]
        # Skip the CLS token (index 0) and any register tokens (indices 1..n_registers).
        # Only the remaining tokens carry spatial patch information.
        n_registers = self.model.config.num_register_tokens
        return embeddings.last_hidden_state[:, 1+n_registers:, :]

    def get_dense_embeddings(self, patch_tokens, target_size):
        n_patches = patch_tokens.shape[1]
        grid_size = int(n_patches ** 0.5)
        embed_dim = patch_tokens.shape[2]
        tokens = patch_tokens[0].reshape(1, grid_size, grid_size, embed_dim)
        tokens = tokens.permute(0, 3, 1, 2)
        dense = F.interpolate(tokens, size=target_size, mode='bilinear', align_corners=False)
        return dense.squeeze(0).permute(1, 2, 0)


def get_all_crops(data_dir: Path, datasets: list[str] = []) -> list[tuple]:
    """Walk through the data directory and get all crops downloaded"""
    crops = []
    if not datasets:
        datasets = [p.name for p in data_dir.iterdir() if p.is_dir()]

    for dataset in datasets:
        dataset_path = data_dir / dataset
        if not dataset_path.is_dir():
            continue
        for crop_path in dataset_path.iterdir():
            if not crop_path.is_dir():
                continue
            crops.append((dataset, crop_path.name, crop_path))
    return crops
