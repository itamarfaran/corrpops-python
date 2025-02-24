import warnings
from pathlib import Path
from typing import Dict, Literal, Union

import numpy as np

from linalg.triangle_vector import vector_to_triangle


def download_data(dst: Path):  # pragma: no cover
    import urllib.request

    src = (
        "https://github.com/itamarfaran/corrpops-python/raw/refs/heads/main/data/"
        + dst.name
    )
    warnings.warn(f"downloading {dst.name} from corrpops-python github repo")
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
        "control": vector_to_triangle(data["control"], diag_value=1.0),
        "diagnosed": vector_to_triangle(data["diagnosed"], diag_value=1.0),
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
