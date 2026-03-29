import argparse
import fsspec
import logging
import numpy as np
import traceback
import zarr

from pathlib import Path
from utils.logging import custom_logging
from utils.io import read_file

custom_logging()
logger = logging.getLogger(__name__)


OUTPUT_DIR = Path("data")


def open_group(s3_path: str) -> zarr.Group:
    store = fsspec.get_mapper(s3_path, anon=True)
    return zarr.open_group(store, mode='r')


def get_metadata(group: zarr.Group, path: str, scale: str):
    """Read translation (offset) and voxel size in nm from multiscales attrs at given path and scale level."""
    datasets = group[path].attrs['multiscales'][0]['datasets']
    scale_meta = next(
        (d for d in datasets if d['path'] == scale), 
        None
    )
    if scale_meta is None:
        raise ValueError(f"Scale '{scale}' not found at path '{path}'")

    scale_dim = scale_meta['coordinateTransformations'][0]['scale']  # [z, y, x] nm
    translation_dim = scale_meta['coordinateTransformations'][1]['translation']  # [z, y, x] nm
    return scale_dim, translation_dim


def find_crops_with_mito(group: zarr.Group) -> list[str]:
    """Return list of crop names that have a mito label."""
    gt = group['recon-1/labels/groundtruth']
    return [
        name for name, crop in gt.groups() 
        if 'mito' in crop.group_keys()
    ]


def get_em_patch_for_crop(
    group: zarr.Group,
    crop_name: str,
    scale: str
) -> tuple[np.ndarray, np.ndarray]:
    """
    Given a crop name, fetch the aligned EM patch and mito mask at the given scale.
    Returns (em_patch, mito_mask) as numpy arrays.
    """
    mito_path = f'recon-1/labels/groundtruth/{crop_name}/mito'
    em_path = 'recon-1/em/fibsem-uint8'

    # get voxel sizes in nm
    em_voxel_size, _ = get_metadata(group, em_path, scale)
    _, crop_translation = get_metadata(group, mito_path, scale)

    # convert nm offset to voxel indices in EM array
    z_start = int(round(crop_translation[0] / em_voxel_size[0]))
    y_start = int(round(crop_translation[1] / em_voxel_size[1]))
    x_start = int(round(crop_translation[2] / em_voxel_size[2]))

    # get mito array — shape tells us the crop size
    mito_arr = group[f'{mito_path}/{scale}']
    z_size, y_size, x_size = mito_arr.shape

    # fetch aligned EM patch — zarr only downloads these chunks from S3
    em_arr = group[f'{em_path}/{scale}']
    em_patch = em_arr[
        z_start:z_start + z_size,
        y_start:y_start + y_size,
        x_start:x_start + x_size
    ]

    # fetch mito mask
    mito_mask = mito_arr[:]

    logger.info(f"  crop {crop_name}: offset=({z_start}, {y_start}, {x_start}) shape=({z_size}, {y_size}, {x_size})")

    assert em_patch.shape == mito_mask.shape, (
        f"Shape mismatch: em={em_patch.shape} mask={mito_mask.shape}"
    )

    return em_patch, mito_mask


def save(dataset_name: str, crop_name: str, em_patch: np.ndarray, mito_mask: np.ndarray):
    out = OUTPUT_DIR / dataset_name / crop_name
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "em.npy", em_patch)
    np.save(out / "mito_mask.npy", mito_mask)
    logger.info(f"  saved em={em_patch.shape} mask={mito_mask.shape} → {out}")


def download_dataset(name: str, s3_path: str, scale: str):
    """
    Download EM patch and mito mask for all crops in a dataset.
    
    Args:
        name:    dataset name, used as output directory name e.g. 'jrc_hela-3'
        s3_path: full S3 path to zarr store e.g. 's3://janelia-cosem-datasets/jrc_hela-3/jrc_hela-3.zarr'
        scale:   resolution level to download, one of 's0'-'s4'. 
                 's2' recommended — balances detail vs download size.
    """
    logger.info(f"\nOpening {name}...")
    group = open_group(s3_path)

    crops = find_crops_with_mito(group)
    logger.info(f"  found {len(crops)} crops with mito: {crops}")

    for crop_name in crops:
        logger.info(f"  processing {crop_name}...")
        try:
            em_patch, mito_mask = get_em_patch_for_crop(group, crop_name, scale)
            save(name, crop_name, em_patch, mito_mask)
        except Exception as e:
            logger.error(f"  skipping {crop_name}: {e}")
            logger.debug(traceback.format_exc())


if __name__ == "__main__":
    OUTPUT_DIR.mkdir(exist_ok=True)
    parser = argparse.ArgumentParser(
        prog='Download Janelia Cosem Datasets',
        description='Download the data from the janelia-cosem-datasets s3 bucket based on the dataset names and scale',
    )

    parser.add_argument(
        '-c', 
        '--config', 
        required=True, 
        help='Path to config file'
    )
    args = parser.parse_args()

    config = read_file(args.config)

    dataset_names = config['DATASET_NAMES']
    s3_path = config['S3_PATH']
    scale = config['SCALE']

    for name in dataset_names:
        dataset_path = f'{s3_path}/{name}/{name}.zarr'
        download_dataset(name, dataset_path, scale)
