import gc
import numpy as np
import psutil
import torch

from PIL import Image
from pathlib import Path
from dinov3 import DinoV3Model, get_all_crops

DATA_DIR = Path("data")


if __name__ == "__main__":

    model = DinoV3Model()

    print(f"[after model load] RAM: {psutil.virtual_memory().used / 1e9:.1f}GB / {psutil.virtual_memory().total / 1e9:.1f}GB")
    print(f"[after model load] GPU: {torch.cuda.memory_allocated() / 1e9:.1f}GB / {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")

    all_crops = get_all_crops(DATA_DIR)
    print(f"Total crops: {len(all_crops)}")

    for dataset, crop, crop_path in all_crops:
        save_path = crop_path / 'mito_embeddings.npz'
        if save_path.exists():
            print(f"  skipping {dataset}/{crop} — already done")
            continue

        print(f"\n{dataset}/{crop}")
        em = np.load(crop_path / 'em.npy', mmap_mode='r')

        slices = []

        with torch.no_grad():
            for z_idx in range(em.shape[0]):
                inp_image = Image.fromarray(em[z_idx]).convert('RGB')
                target_size = (inp_image.size[1], inp_image.size[0])

                embeddings = model.get_embeddings(inp_image)
                patch_tokens = model.get_patch_tokens(embeddings)
                dense = model.get_dense_embeddings(patch_tokens, target_size)  # (H, W, 1024)

                slices.append(dense.detach().cpu().numpy().astype(np.float16))

                del embeddings, patch_tokens, dense
                torch.cuda.empty_cache()

                print(f"  [z={z_idx}/{em.shape[0]-1}] RAM: {psutil.virtual_memory().used / 1e9:.1f}GB")

        volume = np.stack(slices, axis=0)  # (Z, H, W, 1024)
        np.savez_compressed(save_path, embeddings=volume)
        print(f"  saved {volume.shape} → {save_path}")

        del slices, volume
        gc.collect()
        torch.cuda.empty_cache()
