from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from rbsog_md.utils import save_json


def _safe_linear_drift(time: np.ndarray, values: np.ndarray) -> float:
    mask = np.isfinite(time) & np.isfinite(values)
    if int(np.sum(mask)) < 2:
        return float("nan")
    x = time[mask] - time[mask][0]
    y = values[mask]
    slope, _ = np.polyfit(x, y, deg=1)
    return float(slope)


def compute_stability_summary(records: list[dict[str, float]]) -> dict[str, float]:
    if not records:
        return {
            "area_per_lipid_mean": float("nan"),
            "area_per_lipid_std": float("nan"),
            "area_per_lipid_drift_per_time": float("nan"),
            "thickness_proxy_mean": float("nan"),
            "thickness_proxy_std": float("nan"),
            "thickness_proxy_drift_per_time": float("nan"),
        }

    time = np.array([r["time"] for r in records], dtype=float)
    area = np.array([r["area_per_lipid"] for r in records], dtype=float)
    thickness = np.array([r["thickness_proxy"] for r in records], dtype=float)

    return {
        "area_per_lipid_mean": float(np.nanmean(area)),
        "area_per_lipid_std": float(np.nanstd(area)),
        "area_per_lipid_drift_per_time": _safe_linear_drift(time, area),
        "thickness_proxy_mean": float(np.nanmean(thickness)),
        "thickness_proxy_std": float(np.nanstd(thickness)),
        "thickness_proxy_drift_per_time": _safe_linear_drift(time, thickness),
    }


def export_stability_artifacts(
    records: list[dict[str, float]],
    out_dir: Path,
    title: str,
) -> dict[str, float]:
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = compute_stability_summary(records)
    write_stability_series_csv(out_dir / "stability_timeseries.csv", records)
    save_json(out_dir / "stability_summary.json", summary)
    _plot_stability_timeseries(
        records=records,
        path=out_dir / "stability_timeseries.png",
        title=title,
    )
    return summary


def write_stability_series_csv(path: Path, records: list[dict[str, float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["step", "time", "area_per_lipid", "thickness_proxy", "pressure", "temperature"],
        )
        writer.writeheader()
        for row in records:
            writer.writerow(
                {
                    "step": float(row["step"]),
                    "time": float(row["time"]),
                    "area_per_lipid": float(row["area_per_lipid"]),
                    "thickness_proxy": float(row["thickness_proxy"]),
                    "pressure": float(row["pressure"]),
                    "temperature": float(row["temperature"]),
                }
            )


def _plot_stability_timeseries(
    records: list[dict[str, float]],
    path: Path,
    title: str,
) -> None:
    if not records:
        return

    time = np.array([r["time"] for r in records], dtype=float)
    area = np.array([r["area_per_lipid"] for r in records], dtype=float)
    thickness = np.array([r["thickness_proxy"] for r in records], dtype=float)

    fig, axes = plt.subplots(2, 1, figsize=(7.2, 5.4), sharex=True)

    axes[0].plot(time, area, color="#0f766e", linewidth=1.6)
    axes[0].set_ylabel("Area per lipid")
    axes[0].grid(alpha=0.3)

    axes[1].plot(time, thickness, color="#b45309", linewidth=1.6)
    axes[1].set_ylabel("Thickness proxy")
    axes[1].set_xlabel("Time")
    axes[1].grid(alpha=0.3)

    fig.suptitle(title)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)
