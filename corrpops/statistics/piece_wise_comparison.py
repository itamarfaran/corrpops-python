from typing import Dict, Literal

import numpy as np
from scipy import stats

try:
    from statsmodels.stats.multitest import multipletests

    _statsmodels_installed: bool = True
except ModuleNotFoundError:  # pragma: no cover
    _statsmodels_installed: bool = False  # type: ignore

from linalg.triangle_and_vector import triangle_to_vector


# todo: should be part of package?


def piece_wise_comparison(
    control: np.ndarray,
    diagnosed: np.ndarray,
    p_adjust_method: str = "bonferroni",
    alternative: Literal["two-sided", "smaller", "larger"] = "two-sided",
) -> Dict[str, np.ndarray]:
    if control.ndim == 3:
        control = triangle_to_vector(control)
    if diagnosed.ndim == 3:
        diagnosed = triangle_to_vector(diagnosed)

    control_means = control.mean(axis=0, where=~np.isnan(control))
    control_vars = control.var(axis=0, where=~np.isnan(control))
    control_lens = (~np.isnan(control)).sum(0)

    diagnosed_means = diagnosed.mean(axis=0, where=~np.isnan(diagnosed))
    diagnosed_vars = diagnosed.var(axis=0, where=~np.isnan(diagnosed))
    diagnosed_lens = (~np.isnan(diagnosed)).sum(0)

    df = (control_vars / control_lens + diagnosed_vars / diagnosed_lens) ** 2 / (
        (control_vars / control_lens) ** 2 / (control_lens - 1)
        + (diagnosed_vars / diagnosed_lens) ** 2 / (diagnosed_lens - 1)
    )
    t_vals = (control_means - diagnosed_means) / np.sqrt(
        control_vars / control_lens + diagnosed_vars / diagnosed_lens
    )

    if alternative == "smaller":
        p_vals = stats.t.cdf(t_vals, df=df)
    elif alternative == "larger":
        p_vals = stats.t.sf(t_vals, df=df)
    elif alternative == "two-sided":
        p_vals = 2 * stats.t.sf(np.abs(t_vals), df=df)
    else:
        raise ValueError(  # pragma: no cover
            f"alternative should be one of ['two-sided', 'smaller', 'larger'], "
            f"got {alternative} instead"
        )

    if p_adjust_method == "bonferroni":
        p_vals_adjusted = np.minimum(p_vals * p_vals.size, 1.0)
    elif _statsmodels_installed:
        p_vals_adjusted = multipletests(pvals=p_vals, method=p_adjust_method)[1]
    else:
        raise ModuleNotFoundError(  # pragma: no cover
            "missing optional dependency statsmodels. "
            "only p_adjust_method='bonferroni' is supported. "
            "for other methods please install statsmodels."
        )

    return {
        "control_mean": control_means,
        "control_var": control_vars,
        "control_len": control_lens,
        "diagnosed_mean": diagnosed_means,
        "diagnosed_var": diagnosed_vars,
        "diagnosed_len": diagnosed_lens,
        "t_value": t_vals,
        "df": df,
        "p_value": p_vals,
        "p_value_adjusted": p_vals_adjusted,
    }
