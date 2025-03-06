from pathlib import Path
from typing import Dict, Literal, Union

import numpy as np

from corrpops_logger import corrpops_logger
from linalg.triangle_and_vector import vector_to_triangle

logger = corrpops_logger()


def download_data(dst: Path) -> None:  # pragma: no cover
    import urllib.request

    src = (
        "https://github.com/itamarfaran/corrpops-python/raw/refs/heads/main/data/"
        + dst.name
    )
    logger.warning(f"loaders: downloading {dst.name} from corrpops-python github repo")
    if not dst.parent.exists():
        dst.parent.mkdir()
    urllib.request.urlretrieve(src, dst)


def load_data(  # pragma: no cover
    data_name: Literal[
        "nmda_aal",
        "tda_aal",
        "tea_aal",
        "tga_aal",
        "tiavca_aal",
        "tma_aal",
        "tta_aal",
    ]
) -> Dict[str, np.ndarray]:
    dst = Path(__file__).parent.joinpath(".files", data_name).with_suffix(".npz")
    if not dst.exists():
        download_data(dst)
    data = np.load(dst)

    return {
        "header": data["header"],
        "version": data["version"],
        "control": vector_to_triangle(data["control"]),
        "diagnosed": vector_to_triangle(data["diagnosed"]),
    }


def matlab_to_dict(  # pragma: no cover
    path: Union[Path, str],
    array_key: str,
    control_indices_key: str,
    diagnosed_indices_key: str,
) -> Dict[str, Union[str, np.ndarray]]:
    from scipy.io import loadmat

    data = loadmat(path)
    arr = data[array_key].squeeze()
    arr = np.moveaxis(arr, -1, 0)

    control_indices = data[control_indices_key].flatten() - 1
    diagnosed_indices = data[diagnosed_indices_key].flatten() - 1

    return {
        "header": data["__header__"].decode("utf-8"),
        "version": data["__version__"],
        "control": arr[control_indices],
        "diagnosed": arr[diagnosed_indices],
    }
