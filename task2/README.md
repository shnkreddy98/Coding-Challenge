# MIA-AI Take-Home Challenge — Task 2

## Setup

```bash
pip install -r requirements.txt
```

Requires a CUDA-capable GPU (~1.4 GB VRAM for ViT-L, ~7 GB RAM after model load).


## Task 1 — Data Acquisition

Downloads EM image patches and mitochondria masks from the [OpenOrganelle](https://openorganelle.janelia.org) S3 bucket (public, anonymous access).

For each dataset, the script finds all annotated crops that contain actual mitochondria voxels and downloads:
- `em.npy` — raw EM image patch as `(Z, Y, X)` uint8
- `mito_mask.npy` — binary mitochondria mask aligned to the EM patch

### Output structure

```
data/
├── jrc_hela-2/
│   └── crop_X/
│       ├── em.npy
│       └── mito_mask.npy
└── jrc_hela-3/
    └── crop_X/
        ├── em.npy
        └── mito_mask.npy
```

### Config

```yaml
S3_PATH: s3://janelia-cosem-datasets
DATASET_NAMES:
  - jrc_hela-2
  - jrc_hela-3
SCALE: s2
```

Scale `s2` gives 16 nm/px in y/x and 12.96 nm/px in z — a good balance between structural detail and download size.

**Note on resolution mismatch:** mito labels at `s2` have 2× finer voxel size than EM in all dimensions. The download script converts physical nm offsets before computing EM voxel indices so the two arrays align correctly.

---

## Task 2 — Feature Extraction with DINO

**Model:** DINOv3 ViT-L/16 (`facebook/dinov3-vitl16-pretrain-lvd1689m`) via HuggingFace Transformers.

EM slices are greyscale — each slice is converted to RGB by repeating the single channel three times before passing to the processor.

### 1. Patch Size Selection

**Chosen patch size: 16 × 16 pixels.**

At scale `s2` each pixel is 16 nm, so one patch covers **256 nm × 256 nm** of real tissue. Mitochondria are 500 nm – 1 µm in diameter, so 2–4 patches fit across a single mitochondrion. This means each patch sees a coherent sub-organelle structure — outer membrane, cristae, or matrix — without mixing organelles or losing spatial context.

- Smaller patches (p8 = 128 nm): each patch sees only a partial membrane sliver, losing structural context.
- Larger patches (p32 = 512 nm): each patch exceeds mitochondrion diameter, mixing mito content with surrounding cytoplasm.

DINOv3 natively uses p16, which aligns exactly with this reasoning.

### 2. Dense Per-Pixel Embeddings

ViT produces one token per patch — a 14 × 14 grid for 224 × 224 input. To obtain per-pixel embeddings, the 14 × 14 patch token grid is bilinearly interpolated back to the original image size.

```python
# patch_tokens: (1, 196, 1024) — skip CLS and register tokens
tokens = patch_tokens[0].reshape(1, 14, 14, 1024).permute(0, 3, 1, 2)
dense = F.interpolate(tokens, size=target_size, mode='bilinear', align_corners=False)
dense = dense.squeeze(0).permute(1, 2, 0)  # (H, W, 1024)
```

Bilinear interpolation is preferred over nearest-neighbour because it produces smooth transitions at patch boundaries rather than hard block artifacts, which matters when computing per-pixel cosine similarity.

The full dense volume is saved per crop as `(Z, H, W, 1024)` float16 in `mito_embeddings.npz`, allowing any pixel to be queried directly by index.

### Visualisation

`visualize_embeddings.py` is a lightweight Streamlit dashboard that loads one z-slice at a time and shows the EM + mito overlay alongside the dense embeddings reduced to RGB via PCA. The PCA colour structure confirms the off-the-shelf DINO features are already semantically meaningful — mito regions cluster to a distinct colour with no training.

---
