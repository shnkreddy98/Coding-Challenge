## Task 1 - Data Acquisition

Downloads EM image patches and mitochondria masks from the [OpenOrganelle](https://openorganelle.janelia.org) repository.

## What it does

For each dataset, finds all annotated crops containing mitochondria and downloads:
- `em.npy` — raw EM image patch
- `mito_mask.npy` — binary mitochondria mask aligned to the EM patch

## Output structure

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

## Setup

```bash
pip install -r requirements.txt
```

## Usage

```bash
python download.py -c config.yaml
```

## Config

```yaml
# S3 base path — no trailing slash
# full path constructed as {S3_PATH}/{name}/{name}.zarr
S3_PATH: s3://janelia-cosem-datasets

# must match S3 bucket directory names exactly
DATASET_NAMES:
  - jrc_hela-2
  - jrc_hela-3

SCALE: s2
```
