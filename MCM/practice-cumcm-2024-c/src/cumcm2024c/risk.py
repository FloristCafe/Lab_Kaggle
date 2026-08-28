from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .data import CropPlanningData


@dataclass(frozen=True)
class RiskSimulationResult:
    correlation: pd.DataFrame
    prior_covariance: pd.DataFrame
    simulated_returns: pd.DataFrame
    sample_covariance: pd.DataFrame
    cov_dict: dict[tuple[str, str], float]


def simulate_price_covariance(
    data: CropPlanningData,
    *,
    n_samples: int = 1000,
    seed: int = 2024,
) -> RiskSimulationResult:
    crops = data.crops
    categories = {crop: _category(data.crop_type[crop]) for crop in crops}
    corr = _build_category_correlation(crops, categories)
    std = np.array([_annual_return_std(crop, categories[crop]) for crop in crops], dtype=float)
    prior_cov = np.outer(std, std) * corr
    prior_cov = _nearest_positive_semidefinite(prior_cov)

    mean = np.array([_annual_return_mean(crop, categories[crop]) for crop in crops], dtype=float)
    rng = np.random.default_rng(seed)
    samples = rng.multivariate_normal(mean=mean, cov=prior_cov, size=n_samples)

    simulated_returns = pd.DataFrame(samples, columns=crops)
    sample_cov = simulated_returns.cov()
    cov_dict = {
        (j, k): float(sample_cov.loc[j, k])
        for j in crops
        for k in crops
    }

    return RiskSimulationResult(
        correlation=pd.DataFrame(corr, index=crops, columns=crops),
        prior_covariance=pd.DataFrame(prior_cov, index=crops, columns=crops),
        simulated_returns=simulated_returns,
        sample_covariance=sample_cov,
        cov_dict=cov_dict,
    )


def _category(crop_type: str) -> str:
    text = crop_type.lower()
    if "edible fungi" in text:
        return "edible_fungi"
    if "vegetable" in text:
        return "vegetable_legume" if "legume" in text else "vegetable"
    if "grain" in text:
        return "grain_legume" if "legume" in text else "grain"
    return "other"


def _build_category_correlation(crops: list[str], categories: dict[str, str]) -> np.ndarray:
    matrix = np.eye(len(crops), dtype=float)
    for i, crop_i in enumerate(crops):
        for j, crop_j in enumerate(crops):
            if i == j:
                continue
            cat_i = categories[crop_i]
            cat_j = categories[crop_j]
            if cat_i == cat_j:
                corr = 0.70
            elif cat_i.split("_")[0] == cat_j.split("_")[0]:
                corr = 0.55
            elif {"grain", "grain_legume"} & {cat_i, cat_j} and {"vegetable", "vegetable_legume"} & {cat_i, cat_j}:
                corr = -0.10
            elif "edible_fungi" in {cat_i, cat_j}:
                corr = 0.05
            else:
                corr = 0.15
            matrix[i, j] = corr
    return _nearest_correlation_matrix(matrix)


def _annual_return_std(crop: str, category: str) -> float:
    if crop == "Morel":
        return 0.050
    if category == "edible_fungi":
        return 0.040
    if category in {"vegetable", "vegetable_legume"}:
        return 0.050
    if category in {"grain", "grain_legume"}:
        return 0.020
    return 0.030


def _annual_return_mean(crop: str, category: str) -> float:
    if crop == "Morel":
        return -0.050
    if category == "edible_fungi":
        return -0.030
    if category in {"vegetable", "vegetable_legume"}:
        return 0.050
    return 0.0


def _nearest_correlation_matrix(matrix: np.ndarray) -> np.ndarray:
    psd = _nearest_positive_semidefinite(matrix)
    scale = np.sqrt(np.diag(psd))
    corr = psd / np.outer(scale, scale)
    np.fill_diagonal(corr, 1.0)
    return corr


def _nearest_positive_semidefinite(matrix: np.ndarray) -> np.ndarray:
    symmetric = (matrix + matrix.T) / 2.0
    eigvals, eigvecs = np.linalg.eigh(symmetric)
    eigvals = np.clip(eigvals, 1e-10, None)
    return eigvecs @ np.diag(eigvals) @ eigvecs.T
