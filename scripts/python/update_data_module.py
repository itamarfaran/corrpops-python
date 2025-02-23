from pathlib import Path

import numpy as np

from corrpops.data.preproccessing import matlab_to_dict
from corrpops.linalg.triangle_vector import triangle_to_vector


def get_root_directory() -> Path:
    path = Path(__file__)
    while path.name != "corrpops-python":
        path = path.parent
    return path


def update_amnesia(root: Path):
    path = root.joinpath("data", "amnesia_aal_all.mat")

    for key in ["TDA", "TEA", "TGA", "TIACVA", "TMA", "TTA"]:
        results = matlab_to_dict(
            path=path,
            array_key="corrmats",
            control_indices_key="CONTROLS",
            diagnosed_indices_key=key,
        )
        results["control"] = triangle_to_vector(results["control"])
        results["diagnosed"] = triangle_to_vector(results["diagnosed"])
        np.savez_compressed(root.joinpath("data", f"{key.lower()}_aal.npz"), **results)


def update_nmda(root: Path):
    path = root.joinpath("data", "nmda_aal_all.mat")

    results = matlab_to_dict(
        path=path,
        array_key="group_all",
        control_indices_key="CONTROLS",
        diagnosed_indices_key="NMDA",
    )
    results["control"] = triangle_to_vector(results["control"])
    results["diagnosed"] = triangle_to_vector(results["diagnosed"])
    np.savez_compressed(root.joinpath("data", "nmda_aal.npz"), **results)


if __name__ == "__main__":
    root_path = get_root_directory()
    update_amnesia(root_path)
    update_nmda(root_path)
