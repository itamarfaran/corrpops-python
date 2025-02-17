from pathlib import Path
import numpy as np
from utils.triangle_vector import vector_to_triangle


def download_data():
    pass


def load_data(data_name):
    path = Path(__file__).with_name(data_name).with_suffix(".npz")
    if not path.exists():
        download_data()
    data = np.load(path)

    return {
        "header": data["header"],
        "version": data["version"],
        "control": vector_to_triangle(data["control"], diag_value=1.0),
        "diagnosed": vector_to_triangle(data["diagnosed"], diag_value=1.0),
    }


if __name__ == '__main__':
    results = load_data("nmda_aal")
    print("")
