from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass
class BootstrapRatioCI:
    point_estimate: float
    lower: float
    upper: float


@dataclass
class PermutationTestResult:
    observed: float
    p_value: float


@dataclass
class LogLogFitResult:
    slope: float
    intercept: float
    r_squared: float


def bootstrap_mean_ratio_ci(
    numerator: list[float] | np.ndarray,
    denominator: list[float] | np.ndarray,
    *,
    n_resamples: int = 10_000,
    confidence: float = 0.95,
    seed: int = 12345,
) -> BootstrapRatioCI:
    num = np.asarray(numerator, dtype=float)
    den = np.asarray(denominator, dtype=float)
    if num.size == 0 or den.size == 0:
        nan = float("nan")
        return BootstrapRatioCI(point_estimate=nan, lower=nan, upper=nan)

    rng = np.random.default_rng(seed)
    point = float(np.mean(num) / max(float(np.mean(den)), 1e-12))

    samples = np.empty((n_resamples,), dtype=float)
    for i in range(n_resamples):
        num_idx = rng.integers(0, num.size, size=num.size)
        den_idx = rng.integers(0, den.size, size=den.size)
        num_mean = float(np.mean(num[num_idx]))
        den_mean = float(np.mean(den[den_idx]))
        samples[i] = num_mean / max(den_mean, 1e-12)

    alpha = 0.5 * (1.0 - confidence)
    lower = float(np.quantile(samples, alpha))
    upper = float(np.quantile(samples, 1.0 - alpha))
    return BootstrapRatioCI(point_estimate=point, lower=lower, upper=upper)


def permutation_test_mean_difference(
    sample_a: list[float] | np.ndarray,
    sample_b: list[float] | np.ndarray,
    *,
    n_resamples: int = 20_000,
    seed: int = 12345,
) -> PermutationTestResult:
    a = np.asarray(sample_a, dtype=float)
    b = np.asarray(sample_b, dtype=float)
    if a.size == 0 or b.size == 0:
        nan = float("nan")
        return PermutationTestResult(observed=nan, p_value=nan)

    observed = float(np.mean(b) - np.mean(a))
    pooled = np.concatenate([a, b])
    rng = np.random.default_rng(seed)
    exceed = 0
    for _ in range(n_resamples):
        permuted = rng.permutation(pooled)
        diff = float(np.mean(permuted[a.size :]) - np.mean(permuted[: a.size]))
        if abs(diff) >= abs(observed):
            exceed += 1
    p_value = float((exceed + 1) / (n_resamples + 1))
    return PermutationTestResult(observed=observed, p_value=p_value)


def permutation_test_dispersion(
    sample_a: list[float] | np.ndarray,
    sample_b: list[float] | np.ndarray,
    *,
    n_resamples: int = 20_000,
    seed: int = 12345,
) -> PermutationTestResult:
    a = np.asarray(sample_a, dtype=float)
    b = np.asarray(sample_b, dtype=float)
    if a.size == 0 or b.size == 0:
        nan = float("nan")
        return PermutationTestResult(observed=nan, p_value=nan)

    dev_a = np.abs(a - np.median(a))
    dev_b = np.abs(b - np.median(b))
    observed = float(np.mean(dev_b) - np.mean(dev_a))
    pooled = np.concatenate([dev_a, dev_b])
    rng = np.random.default_rng(seed)
    exceed = 0
    for _ in range(n_resamples):
        permuted = rng.permutation(pooled)
        diff = float(np.mean(permuted[dev_a.size :]) - np.mean(permuted[: dev_a.size]))
        if abs(diff) >= abs(observed):
            exceed += 1
    p_value = float((exceed + 1) / (n_resamples + 1))
    return PermutationTestResult(observed=observed, p_value=p_value)


def loglog_fit(x_values: list[float] | np.ndarray, y_values: list[float] | np.ndarray) -> LogLogFitResult:
    x = np.asarray(x_values, dtype=float)
    y = np.asarray(y_values, dtype=float)
    mask = (x > 0.0) & (y > 0.0) & np.isfinite(x) & np.isfinite(y)
    if int(np.sum(mask)) < 2:
        nan = float("nan")
        return LogLogFitResult(slope=nan, intercept=nan, r_squared=nan)

    lx = np.log(x[mask])
    ly = np.log(y[mask])
    slope, intercept = np.polyfit(lx, ly, deg=1)
    pred = slope * lx + intercept
    residual = float(np.sum((ly - pred) ** 2))
    total = float(np.sum((ly - float(np.mean(ly))) ** 2))
    r_squared = float(1.0 - residual / total) if total > 0.0 else 1.0
    return LogLogFitResult(slope=float(slope), intercept=float(intercept), r_squared=r_squared)


def rolling_mean(values: list[float] | np.ndarray, window: int) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if window <= 1 or arr.size == 0:
        return arr.copy()
    out = np.empty_like(arr)
    for idx in range(arr.size):
        left = max(0, idx - window + 1)
        out[idx] = float(np.mean(arr[left : idx + 1]))
    return out


def rolling_std(values: list[float] | np.ndarray, window: int) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if window <= 1 or arr.size == 0:
        return np.zeros_like(arr)
    out = np.empty_like(arr)
    for idx in range(arr.size):
        left = max(0, idx - window + 1)
        chunk = arr[left : idx + 1]
        out[idx] = float(np.std(chunk))
    return out


def mean_and_std(values: list[float] | np.ndarray) -> tuple[float, float]:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        nan = float("nan")
        return (nan, nan)
    return (float(np.mean(arr)), float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0)


def ci95_half_width(values: list[float] | np.ndarray) -> float:
    arr = np.asarray(values, dtype=float)
    if arr.size <= 1:
        return float("nan")

    tcrit = 1.96 if arr.size > 30 else {
        2: 12.706,
        3: 4.303,
        4: 3.182,
        5: 2.776,
        6: 2.571,
        7: 2.447,
        8: 2.365,
        9: 2.306,
        10: 2.262,
        11: 2.228,
        12: 2.201,
        13: 2.179,
        14: 2.160,
        15: 2.145,
        16: 2.131,
        17: 2.120,
        18: 2.110,
        19: 2.101,
        20: 2.093,
        21: 2.086,
        22: 2.080,
        23: 2.074,
        24: 2.069,
        25: 2.064,
        26: 2.060,
        27: 2.056,
        28: 2.052,
        29: 2.048,
        30: 2.045,
    }.get(int(arr.size), 1.96)
    std = float(np.std(arr, ddof=1))
    return float(tcrit * std / math.sqrt(float(arr.size)))
