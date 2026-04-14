import yaml
from pathlib import Path

_CONFIG_PATH = Path(__file__).parent.parent / "config.yaml"

with open(_CONFIG_PATH) as f:
    cfg = yaml.safe_load(f)


def model_tag() -> str:
    """Short identifier derived from model name + proj dim.
    Used to version artifact filenames so different model configs don't collide.
    e.g. 'dinov3-vitl16_256'
    """
    slug = cfg["MODEL_NAME"].split("/")[-1].split("-pretrain")[0]
    return f"{slug}_{cfg['PROJ_DIM']}"
