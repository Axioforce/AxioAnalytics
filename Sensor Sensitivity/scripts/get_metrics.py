import os
import re
import glob
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
from pathlib import Path

# Reuse pipeline helpers from overlay module
from overlay_mounted_vs_reseated import (
    Params,
    load_group,
    preprocess_signal,
    align_group_by_peak,
    align_group_by_local_peak,
    align_run,
    rotate_group_about_z,
)


# -------- Configuration --------
DEFAULT_ALIGNMENT = "local_peak"        # one of: "frame", "event", "peak", "local_peak"
DEFAULT_ROTATION_DEG_Z = -45.0           # rotation applied about Z after alignment
DEFAULT_Z_SMOOTH_WINDOW = 10              # frames for bz smoothing before negative-slope cutoff
DEFAULT_Z_MIN_BZ_FOR_CUTOFF_SEARCH = 300.0  # start looking for negative slope only after bz >= this


@dataclass
class GMPaths:
    project_root: Path = Path(__file__).resolve().parent.parent
    runs_root: Path = project_root / "Load_Cell_Runs"
    metrics_root: Path = project_root / "outputs" / "metrics" / "sensitivity"
    metrics_bz_root: Path = project_root / "outputs" / "metrics" / "Bz_min_max"


def _ensure_dir(path: Path) -> None:
    try:
        path.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass


def _scan_cells(root: Path) -> Dict[str, str]:
    cells: Dict[str, str] = {}
    for p in sorted(root.glob("*/")):
        cells[p.name] = str(p / "*.csv")
    return cells


def _parse_date_from_filename(filename: str) -> Optional[str]:
    base = os.path.basename(filename)
    name, _ = os.path.splitext(base)
    parts = name.split("_")
    if not parts:
        return None
    candidate = parts[-1]
    if re.fullmatch(r"\d{1,2}\.\d{1,2}\.\d{4}", candidate):
        return candidate
    # fallback: look anywhere in the name for last date-like token
    matches = re.findall(r"\d{1,2}\.\d{1,2}\.\d{4}", name)
    return matches[-1] if matches else None


def _linear_slope(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    m = min(x.size, y.size)
    if m < 2:
        return float("nan")
    x = x[:m]
    y = y[:m]
    mask = np.isfinite(x) & np.isfinite(y)
    x2 = x[mask]
    y2 = y[mask]
    if x2.size < 2:
        return float("nan")
    try:
        slope = float(np.polyfit(x2, y2, deg=1)[0])
    except Exception:
        slope = float("nan")
    return slope


def _fit_stats(x: np.ndarray, y: np.ndarray) -> Tuple[float, float, float]:
    """Return (slope, r2, rmse) for y ~ slope*x + intercept using OLS.

    Falls back to NaNs if insufficient data.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    m = min(x.size, y.size)
    if m < 2:
        return float("nan"), float("nan"), float("nan")
    x = x[:m]
    y = y[:m]
    mask = np.isfinite(x) & np.isfinite(y)
    x2 = x[mask]
    y2 = y[mask]
    if x2.size < 2:
        return float("nan"), float("nan"), float("nan")
    try:
        slope, intercept = np.polyfit(x2, y2, deg=1)
        yhat = slope * x2 + intercept
        ss_res = float(np.sum((y2 - yhat) ** 2))
        ss_tot = float(np.sum((y2 - float(np.mean(y2))) ** 2))
        r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")
        rmse = float(np.sqrt(ss_res / max(1, (x2.size - 2))))
        return float(slope), r2, rmse
    except Exception:
        return float("nan"), float("nan"), float("nan")


def _moving_average(y: np.ndarray, window: int) -> np.ndarray:
    if y.size == 0:
        return y
    w = max(1, int(window))
    kernel = np.ones(w, dtype=float) / float(w)
    return np.convolve(y.astype(float), kernel, mode="same")


def _first_negative_slope_index_after_bz_threshold(bz: np.ndarray, min_bz: float, smooth_window: int) -> Optional[int]:
    if bz.size == 0:
        return None
    # Find where raw bz first reaches the threshold
    start_idx = None
    for i in range(bz.size):
        v = bz[i]
        if np.isfinite(v) and (v >= float(min_bz)):
            start_idx = i
            break
    if start_idx is None:
        return None
    # Smooth only for detecting the first negative slope
    ys = _moving_average(bz, smooth_window)
    dy = np.diff(ys)
    scan_from = max(1, int(start_idx))
    for i in range(scan_from, dy.size):
        v = dy[i]
        if np.isfinite(v) and (v < 0):
            return i + 1
    return None


def _preprocess_runs_for_cell(glob_pattern: str, params: Params) -> List[pd.DataFrame]:
    try:
        dfs = load_group(glob_pattern, params.axis_mapping)
    except Exception:
        dfs = []
    if not dfs:
        return []
    try:
        dfs_prep = [preprocess_signal(df, params) for df in dfs]
        if params.alignment_method == "peak":
            dfs_aligned = align_group_by_peak(dfs_prep, params, signal="mag")
        elif params.alignment_method == "local_peak":
            dfs_aligned = align_group_by_local_peak(dfs_prep, params)
        else:
            dfs_aligned = [align_run(df, params) for df in dfs_prep]
        dfs_rot = rotate_group_about_z(dfs_aligned, params.rotation_deg_z)
        return dfs_rot
    except Exception:
        return []


def compute_sensitivity_for_run(
    df: pd.DataFrame,
    z_cut_smooth_window: int,
    min_bz_for_cutoff: float,
) -> Tuple[float, float, float, Optional[int], Optional[float], int, int, int, float, float, float, float, float, float, Optional[float], Optional[float]]:
    slope_x = float("nan")
    slope_y = float("nan")
    slope_z = float("nan")
    r2_x = float("nan")
    r2_y = float("nan")
    r2_z = float("nan")
    rmse_x = float("nan")
    rmse_y = float("nan")
    rmse_z = float("nan")
    cut_idx: Optional[int] = None
    cut_bz: Optional[float] = None
    z_min: Optional[float] = None
    z_max: Optional[float] = None
    n_x = n_y = n_z = 0

    # x, y slopes: measured vs truth
    if ("x" in df.columns) and ("bx" in df.columns):
        n_x = int(min(df["x"].size, df["bx"].size))
        slope_x, r2_x, rmse_x = _fit_stats(df["bx"].to_numpy(), df["x"].to_numpy())
    if ("y" in df.columns) and ("by" in df.columns):
        n_y = int(min(df["y"].size, df["by"].size))
        slope_y, r2_y, rmse_y = _fit_stats(df["by"].to_numpy(), df["y"].to_numpy())

    # z slope with cutoff based on truth bz smoothed slope turning negative (search starts after bz >= threshold)
    if ("z" in df.columns) and ("bz" in df.columns):
        bz = df["bz"].to_numpy(dtype=float)
        # Compute measured sum-z extrema over the FULL run (not truncated)
        try:
            z_all = df["z"].to_numpy(dtype=float)
            if z_all.size:
                z_min = float(np.nanmin(z_all))
                z_max = float(np.nanmax(z_all))
        except Exception:
            pass
        idx = _first_negative_slope_index_after_bz_threshold(bz, float(min_bz_for_cutoff), z_cut_smooth_window)
        if idx is None:
            # If no negative slope found, use full run
            idx_use = bz.size
        else:
            idx_use = max(1, int(idx))
            cut_idx = int(idx)
            try:
                cut_bz = float(bz[cut_idx])
            except Exception:
                cut_bz = None
        z_use = df["z"].to_numpy(dtype=float)[:idx_use]
        bz_use = bz[:idx_use]
        n_z = int(min(z_use.size, bz_use.size))
        slope_z, r2_z, rmse_z = _fit_stats(bz_use, z_use)
        # Note: bz_min/bz_max are intentionally from the full run

    return slope_x, slope_y, slope_z, cut_idx, cut_bz, n_x, n_y, n_z, r2_x, r2_y, r2_z, rmse_x, rmse_y, rmse_z, z_min, z_max


def run_get_metrics(
    runs_root: Path,
    out_dir: Path,
    alignment: str,
    rotation_deg_z: float,
    z_cut_smooth_window: int,
    min_bz_for_cutoff: float,
) -> Dict[str, Path]:
    paths = GMPaths(project_root=runs_root.parent, runs_root=runs_root, metrics_root=out_dir)
    params = Params()
    params.alignment_method = alignment
    params.rotation_deg_z = float(rotation_deg_z)

    cells = _scan_cells(paths.runs_root)
    if not cells:
        return {}

    rows_runs: List[Dict[str, object]] = []
    for cell_name, gpat in cells.items():
        # Load with preprocessing
        dfs = _preprocess_runs_for_cell(gpat, params)
        for df in dfs:
            src = str(df["source_path"].iloc[0]) if "source_path" in df.columns and len(df) > 0 else ""
            date_str = _parse_date_from_filename(src) or ""
            (
                sx, sy, sz,
                cut_idx, cut_bz,
                nx, ny, nz,
                r2x, r2y, r2z,
                rmsex, rmsey, rmsez,
                zmin, zmax,
            ) = compute_sensitivity_for_run(df, z_cut_smooth_window, min_bz_for_cutoff)
            rows_runs.append({
                "cell": cell_name,
                "date": date_str,
                "run_file": src,
                "slope_x": sx,
                "slope_y": sy,
                "slope_z": sz,
                "r2_x": r2x,
                "r2_y": r2y,
                "r2_z": r2z,
                "rmse_x": rmsex,
                "rmse_y": rmsey,
                "rmse_z": rmsez,
                "z_min": zmin if zmin is not None else np.nan,
                "z_max": zmax if zmax is not None else np.nan,
                "z_cut_frame": cut_idx if cut_idx is not None else np.nan,
                "z_cut_bz": cut_bz if cut_bz is not None else np.nan,
                "n_x": nx,
                "n_y": ny,
                "n_z": nz,
            })

    if not rows_runs:
        return {}

    _ensure_dir(paths.metrics_root)
    _ensure_dir(paths.metrics_bz_root)

    df_runs = pd.DataFrame(rows_runs)
    out_paths: Dict[str, Path] = {}
    run_csv = paths.metrics_root / "runs.csv"
    df_runs.to_csv(run_csv, index=False)
    out_paths["runs"] = run_csv

    # Bz_min_max outputs in a separate folder mirroring sensitivity breakdown
    bz_runs = df_runs[["cell", "date", "run_file", "z_min", "z_max", "z_cut_frame", "z_cut_bz"]].copy()
    try:
        bz_runs["z_range"] = bz_runs["z_max"] - bz_runs["z_min"]
    except Exception:
        bz_runs["z_range"] = np.nan
    bz_runs_csv = paths.metrics_bz_root / "runs.csv"
    bz_runs.to_csv(bz_runs_csv, index=False)
    out_paths["bz_runs"] = bz_runs_csv

    # Per cell, per day aggregation for Bz min/max
    if not bz_runs.empty:
        bz_per_day = bz_runs.groupby(["cell", "date"]).agg(
            mean_z_min=("z_min", "mean"),
            sd_z_min=("z_min", "std"),
            mean_z_max=("z_max", "mean"),
            sd_z_max=("z_max", "std"),
            mean_z_range=("z_range", "mean"),
            sd_z_range=("z_range", "std"),
            runs=("run_file", "count"),
        ).reset_index()
        bz_per_day_csv = paths.metrics_bz_root / "per_cell_day.csv"
        bz_per_day.to_csv(bz_per_day_csv, index=False)
        out_paths["bz_per_cell_day"] = bz_per_day_csv

    # Per cell overall aggregation for Bz min/max
    bz_per_cell = bz_runs.groupby(["cell"]).agg(
        mean_z_min=("z_min", "mean"),
        sd_z_min=("z_min", "std"),
        mean_z_max=("z_max", "mean"),
        sd_z_max=("z_max", "std"),
        mean_z_range=("z_range", "mean"),
        sd_z_range=("z_range", "std"),
        runs=("run_file", "count"),
        days=("date", pd.Series.nunique),
    ).reset_index()
    bz_per_cell_csv = paths.metrics_bz_root / "per_cell_overall.csv"
    bz_per_cell.to_csv(bz_per_cell_csv, index=False)
    out_paths["bz_per_cell_overall"] = bz_per_cell_csv

    # Fleet comparison for Bz min/max
    fleet_bz = {
        "fleet_mean_z_min": float(bz_per_cell["mean_z_min"].mean()) if not bz_per_cell.empty else np.nan,
        "fleet_mean_z_max": float(bz_per_cell["mean_z_max"].mean()) if not bz_per_cell.empty else np.nan,
        "fleet_sd_z_min": float(bz_per_cell["sd_z_min"].mean()) if not bz_per_cell.empty else np.nan,
        "fleet_sd_z_max": float(bz_per_cell["sd_z_max"].mean()) if not bz_per_cell.empty else np.nan,
        "fleet_mean_z_range": float(bz_per_cell["mean_z_range"].mean()) if not bz_per_cell.empty else np.nan,
        "fleet_sd_z_range": float(bz_per_cell["sd_z_range"].mean()) if not bz_per_cell.empty else np.nan,
    }
    bz_comp_rows: List[Dict[str, object]] = []
    for _, row in bz_per_cell.iterrows():
        bz_comp_rows.append({
            "cell": row["cell"],
            "mean_z_min": row["mean_z_min"],
            "mean_z_max": row["mean_z_max"],
            "sd_z_min": row["sd_z_min"],
            "sd_z_max": row["sd_z_max"],
            "mean_z_range": row["mean_z_range"],
            "sd_z_range": row["sd_z_range"],
            "delta_z_min": row["mean_z_min"] - fleet_bz["fleet_mean_z_min"],
            "delta_z_max": row["mean_z_max"] - fleet_bz["fleet_mean_z_max"],
            "delta_z_range": row["mean_z_range"] - fleet_bz["fleet_mean_z_range"],
            "ratio_z_min": row["mean_z_min"] / fleet_bz["fleet_mean_z_min"] if np.isfinite(fleet_bz["fleet_mean_z_min"]) and fleet_bz["fleet_mean_z_min"] != 0 else np.nan,
            "ratio_z_max": row["mean_z_max"] / fleet_bz["fleet_mean_z_max"] if np.isfinite(fleet_bz["fleet_mean_z_max"]) and fleet_bz["fleet_mean_z_max"] != 0 else np.nan,
            "ratio_z_range": row["mean_z_range"] / fleet_bz["fleet_mean_z_range"] if np.isfinite(fleet_bz["fleet_mean_z_range"]) and fleet_bz["fleet_mean_z_range"] != 0 else np.nan,
            **fleet_bz,
        })
    bz_comp_df = pd.DataFrame(bz_comp_rows)
    bz_comp_csv = paths.metrics_bz_root / "fleet_comparison.csv"
    bz_comp_df.to_csv(bz_comp_csv, index=False)
    out_paths["bz_fleet_comparison"] = bz_comp_csv

    # Single Cell, Within a Set (same day): aggregate per cell/date
    if not df_runs.empty:
        grp_cols = ["cell", "date"]
        per_day = df_runs.groupby(grp_cols).agg(
            mean_slope_x=("slope_x", "mean"),
            sd_slope_x=("slope_x", "std"),
            mean_slope_y=("slope_y", "mean"),
            sd_slope_y=("slope_y", "std"),
            mean_slope_z=("slope_z", "mean"),
            sd_slope_z=("slope_z", "std"),
            mean_r2_x=("r2_x", "mean"),
            sd_r2_x=("r2_x", "std"),
            mean_r2_y=("r2_y", "mean"),
            sd_r2_y=("r2_y", "std"),
            mean_r2_z=("r2_z", "mean"),
            sd_r2_z=("r2_z", "std"),
            mean_rmse_x=("rmse_x", "mean"),
            sd_rmse_x=("rmse_x", "std"),
            mean_rmse_y=("rmse_y", "mean"),
            sd_rmse_y=("rmse_y", "std"),
            mean_rmse_z=("rmse_z", "mean"),
            sd_rmse_z=("rmse_z", "std"),
            runs=("run_file", "count"),
        ).reset_index()
        per_day_csv = paths.metrics_root / "per_cell_day.csv"
        per_day.to_csv(per_day_csv, index=False)
        out_paths["per_cell_day"] = per_day_csv

    # Single Cell, Across Sets (all days): aggregate per cell
    per_cell = df_runs.groupby(["cell"]).agg(
        mean_slope_x=("slope_x", "mean"),
        sd_slope_x=("slope_x", "std"),
        mean_slope_y=("slope_y", "mean"),
        sd_slope_y=("slope_y", "std"),
        mean_slope_z=("slope_z", "mean"),
        sd_slope_z=("slope_z", "std"),
        mean_r2_x=("r2_x", "mean"),
        sd_r2_x=("r2_x", "std"),
        mean_r2_y=("r2_y", "mean"),
        sd_r2_y=("r2_y", "std"),
        mean_r2_z=("r2_z", "mean"),
        sd_r2_z=("r2_z", "std"),
        mean_rmse_x=("rmse_x", "mean"),
        sd_rmse_x=("rmse_x", "std"),
        mean_rmse_y=("rmse_y", "mean"),
        sd_rmse_y=("rmse_y", "std"),
        mean_rmse_z=("rmse_z", "mean"),
        sd_rmse_z=("rmse_z", "std"),
        runs=("run_file", "count"),
        days=("date", pd.Series.nunique),
    ).reset_index()
    per_cell_csv = paths.metrics_root / "per_cell_overall.csv"
    per_cell.to_csv(per_cell_csv, index=False)
    out_paths["per_cell_overall"] = per_cell_csv

    # Fleet-level average and per-cell deltas
    fleet = {
        "fleet_mean_x": float(per_cell["mean_slope_x"].mean()) if not per_cell.empty else np.nan,
        "fleet_mean_y": float(per_cell["mean_slope_y"].mean()) if not per_cell.empty else np.nan,
        "fleet_mean_z": float(per_cell["mean_slope_z"].mean()) if not per_cell.empty else np.nan,
    }
    comp_rows: List[Dict[str, object]] = []
    for _, row in per_cell.iterrows():
        comp_rows.append({
            "cell": row["cell"],
            "mean_slope_x": row["mean_slope_x"],
            "mean_slope_y": row["mean_slope_y"],
            "mean_slope_z": row["mean_slope_z"],
            "delta_x": row["mean_slope_x"] - fleet["fleet_mean_x"],
            "delta_y": row["mean_slope_y"] - fleet["fleet_mean_y"],
            "delta_z": row["mean_slope_z"] - fleet["fleet_mean_z"],
            "ratio_x": row["mean_slope_x"] / fleet["fleet_mean_x"] if np.isfinite(fleet["fleet_mean_x"]) and fleet["fleet_mean_x"] != 0 else np.nan,
            "ratio_y": row["mean_slope_y"] / fleet["fleet_mean_y"] if np.isfinite(fleet["fleet_mean_y"]) and fleet["fleet_mean_y"] != 0 else np.nan,
            "ratio_z": row["mean_slope_z"] / fleet["fleet_mean_z"] if np.isfinite(fleet["fleet_mean_z"]) and fleet["fleet_mean_z"] != 0 else np.nan,
            **fleet,
        })
    comp_df = pd.DataFrame(comp_rows)
    comp_csv = paths.metrics_root / "fleet_comparison.csv"
    comp_df.to_csv(comp_csv, index=False)
    out_paths["fleet_comparison"] = comp_csv

    return out_paths


def main() -> None:
    gm_paths = GMPaths()
    out_paths = run_get_metrics(
        runs_root=gm_paths.runs_root,
        out_dir=gm_paths.metrics_root,
        alignment=DEFAULT_ALIGNMENT,
        rotation_deg_z=DEFAULT_ROTATION_DEG_Z,
        z_cut_smooth_window=DEFAULT_Z_SMOOTH_WINDOW,
        min_bz_for_cutoff=DEFAULT_Z_MIN_BZ_FOR_CUTOFF_SEARCH,
    )
    if out_paths:
        print("[get_metrics] Wrote:")
        for k, p in out_paths.items():
            print(f"  - {k}: {p}")
    else:
        print("[get_metrics] No data found or nothing written.")


if __name__ == "__main__":
    main()


