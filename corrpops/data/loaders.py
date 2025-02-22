import warnings
from pathlib import Path
from typing import Dict, Literal

import numpy as np

from linalg.triangle_vector import vector_to_triangle


def download_data(dst: Path):
    import urllib.request

    src = (
        "https://github.com/itamarfaran/corrpops-python/raw/refs/heads/main/data/"
        + dst.name
    )
    warnings.warn(f"downloading {dst.name} from corrpops-python github repo")
    urllib.request.urlretrieve(src, dst)


def load_data(
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
    path = Path(__file__).parent.with_name(data_name).with_suffix(".npz")
    if not path.exists():
        download_data(path)
    data = np.load(path)

    return {
        "header": data["header"],
        "version": data["version"],
        "control": vector_to_triangle(data["control"], diag_value=1.0),
        "diagnosed": vector_to_triangle(data["diagnosed"], diag_value=1.0),
    }
