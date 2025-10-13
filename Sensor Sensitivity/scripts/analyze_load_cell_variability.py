import os
import glob
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.widgets import CheckButtons
from pathlib import Path


@dataclass
class Paths:
    # Resolve paths relative to the project folder (one level above this script)
    _project_root: Path = Path(__file__).resolve().parent.parent
    mounted_glob: str = str(_project_root / "Load_Cell_Spiral_test" / "H6b.90" / "10.08.2025" / "mounted_runs" / "*.csv")
    reseated_glob: str = str(_project_root / "Load_Cell_Spiral_test" / "H6b.90" / "10.08.2025" / "stationary_runs" / "*.csv")
    plots_dir: str = str(_project_root / "outputs" / "plots")
    metrics_dir: str = str(_project_root / "outputs" / "metrics")


@dataclass
class Params:
    axis_mapping: Dict[str, str] = None  # to be filled in __post_init__
    baseline_seconds: float = 0.5  # seconds for baseline window
    apply_filter: bool = False
    filter_cutoff_hz: float = 20.0
    filter_order: int = 2
    alignment_method: str = "local_peak"  # "frame" | "event" | "peak" | "local_peak"
    alignment_signal: str = "mag"  # which signal to detect peak on: "mag" or "bmag"
    onset_threshold: float = 0.05  # threshold on magnitude after baseline removal
    plot_mean_band: bool = True
    local_peak_threshold: float = 1000.0  # threshold for local peak window on 'x'
    rotation_deg_z: float =  -45.0  # rotate measured x/y/z about Z after alignment

    def __post_init__(self) -> None:
        if self.axis_mapping is None:
            # Default to summed axes. Change to e.g. '1-inner-x' if you need a specific cell.
            self.axis_mapping = {"x": "sum-x", "y": "sum-y", "z": "sum-z"}


def ensure_dirs(paths: Paths) -> None:
    os.makedirs(paths.plots_dir, exist_ok=True)
    os.makedirs(paths.metrics_dir, exist_ok=True)


def load_group(file_glob: str, axis_mapping: Dict[str, str]) -> List[pd.DataFrame]:
    dataframes: List[pd.DataFrame] = []
    for path in sorted(glob.glob(file_glob)):
        df = pd.read_csv(path)
        # Build a tidy frame with only required columns
        keep_cols = ["time", "record_id"]
        for key in ("x", "y", "z"):
            col = axis_mapping[key]
            if col not in df.columns:
                raise ValueError(f"Column '{col}' not found in {path}")
            keep_cols.append(col)

        # Add truth columns if available
        truth_cols = [c for c in ("bx", "by", "bz") if c in df.columns]
        keep_cols.extend(truth_cols)

        present_cols = [c for c in keep_cols if c in df.columns]
        tidy = df[present_cols].copy()
        # Compute magnitude
        tidy.rename(columns={axis_mapping["x"]: "x", axis_mapping["y"]: "y", axis_mapping["z"]: "z"}, inplace=True)
        tidy["mag"] = np.sqrt(tidy["x"] ** 2 + tidy["y"] ** 2 + tidy["z"] ** 2)
        # Compute truth magnitude if truth exists
        if all(c in tidy.columns for c in ("bx", "by", "bz")):
            tidy["bmag"] = np.sqrt(tidy["bx"] ** 2 + tidy["by"] ** 2 + tidy["bz"] ** 2)
        tidy["source_path"] = os.path.basename(path)
        dataframes.append(tidy)
    return dataframes


def estimate_sampling_rate_hz(df: pd.DataFrame) -> float:
    if "time" not in df.columns:
        # Fallback to index as frames with unknown rate
        return np.nan
    # time appears to be in ms; infer dt from median diff
    t = df["time"].to_numpy()
    if len(t) < 2:
        return np.nan
    dt_ms = np.median(np.diff(t))
    if dt_ms <= 0:
        return np.nan
    return 1000.0 / dt_ms


def butter_lowpass(cutoff_hz: float, fs_hz: float, order: int = 2) -> Tuple[np.ndarray, np.ndarray]:
    nyq = 0.5 * fs_hz
    normal_cutoff = cutoff_hz / nyq
    b, a = butter(order, normal_cutoff, btype="low", analog=False)
    return b, a


def preprocess_signal(df: pd.DataFrame, params: Params) -> pd.DataFrame:
    out = df.copy()
    # Baseline: using first N seconds if time exists, else first N samples as approx
    if "time" in out.columns:
        t0 = out["time"].iloc[0]
        baseline_mask = out["time"] <= (t0 + params.baseline_seconds * 1000.0)
    else:
        n = max(1, int(0.5 * len(out)))  # fallback: 50% of start; conservative
        baseline_mask = pd.Series([True] * n + [False] * (len(out) - n))

    for col in ("x", "y", "z", "mag", "bx", "by", "bz", "bmag"):
        baseline = out.loc[baseline_mask, col].median()
        if col in out.columns:
            out[col] = out[col] - baseline

    if params.apply_filter:
        fs_hz = estimate_sampling_rate_hz(out)
        if not np.isnan(fs_hz) and fs_hz > 0 and params.filter_cutoff_hz < 0.5 * fs_hz:
            b, a = butter_lowpass(params.filter_cutoff_hz, fs_hz, params.filter_order)
            for col in ("x", "y", "z", "mag"):
                out[col] = filtfilt(b, a, out[col].to_numpy(), method="gust")
    return out


def find_onset_index(mag: np.ndarray, threshold: float) -> int:
    idx = np.argmax(mag >= threshold)
    # If no crossing, return 0
    return int(idx) if mag[idx] >= threshold else 0


def align_run(df: pd.DataFrame, params: Params) -> pd.DataFrame:
    if params.alignment_method == "frame":
        # Start at frame 0; nothing to shift
        return df.reset_index(drop=True)

    if params.alignment_method == "event":
        onset = find_onset_index(df["mag"].to_numpy(), params.onset_threshold)
        if onset <= 0:
            return df.reset_index(drop=True)
        # Shift so onset becomes index 0
        aligned = df.iloc[onset:].reset_index(drop=True).copy()
        # Optionally add a relative time column
        if "time" in aligned.columns:
            t0 = aligned["time"].iloc[0]
            aligned["time_rel_ms"] = aligned["time"] - t0
        return aligned

    raise ValueError(f"Unknown alignment_method: {params.alignment_method}")


def align_group_by_peak(dfs: List[pd.DataFrame], params: Params, signal: Optional[str] = None) -> List[pd.DataFrame]:
    """Align a group of runs by peak index while preserving full window.

    Adds a "frame_aligned" column so that peaks coincide across runs.
    No rows are dropped; instead, each run gets its own aligned x-axis.
    """
    if not dfs:
        return dfs
    sig = signal if signal is not None else params.alignment_signal
    # Choose column to use
    if sig not in dfs[0].columns:
        # Fallback to mag if requested signal missing
        sig = "mag" if "mag" in dfs[0].columns else list(dfs[0].columns)[0]

    peak_indices: List[int] = []
    for df in dfs:
        y = df[sig].to_numpy()
        if y.size == 0:
            peak_indices.append(0)
        else:
            peak_indices.append(int(np.nanargmax(np.abs(y))))

    # Reference as median peak for stability
    ref_peak = int(np.median(peak_indices))

    aligned: List[pd.DataFrame] = []
    for df, p in zip(dfs, peak_indices):
        n = len(df)
        aligned_x = np.arange(n) - p + ref_peak
        out = df.copy()
        out["frame_aligned"] = aligned_x
        aligned.append(out)
    return aligned


def find_local_peak_window_index(x: np.ndarray, threshold: float) -> Optional[int]:
    """Return index of the peak within the first window where x crosses above threshold and back below.

    If no such window exists, return None.
    """
    above = x > threshold
    if not np.any(above):
        return None
    # Find first rising edge
    start_idx = None
    for i in range(1, len(x)):
        if not above[i - 1] and above[i]:
            start_idx = i
            break
    if start_idx is None:
        return None
    # Find first falling edge after start
    end_idx = None
    for j in range(start_idx + 1, len(x)):
        if above[j - 1] and not above[j]:
            end_idx = j
            break
    if end_idx is None:
        return None
    # Peak within [start_idx, end_idx)
    if end_idx <= start_idx:
        return None
    window = x[start_idx:end_idx]
    peak_rel = int(np.nanargmax(window))
    return start_idx + peak_rel


def align_group_by_local_peak(dfs: List[pd.DataFrame], params: Params, threshold: Optional[float] = None) -> List[pd.DataFrame]:
    """Align runs by the first local peak in 'x' within the window crossing a threshold.

    Preserves full windows; adds 'frame_aligned'.
    """
    if not dfs:
        return dfs
    thr = params.local_peak_threshold if threshold is None else threshold
    peak_indices: List[int] = []
    for df in dfs:
        x = df["x"].to_numpy()
        idx = find_local_peak_window_index(x, thr)
        if idx is None:
            # fallback to global |x| peak
            idx = int(np.nanargmax(np.abs(x)))
        peak_indices.append(int(idx))

    ref_peak = int(np.median(peak_indices))
    aligned: List[pd.DataFrame] = []
    for df, p in zip(dfs, peak_indices):
        n = len(df)
        aligned_x = np.arange(n) - p + ref_peak
        out = df.copy()
        out["frame_aligned"] = aligned_x
        aligned.append(out)
    return aligned


def trim_to_common_length(groups: List[List[pd.DataFrame]]) -> List[List[pd.DataFrame]]:
    min_len = None
    for group in groups:
        for df in group:
            n = len(df)
            if min_len is None or n < min_len:
                min_len = n
    if min_len is None or min_len <= 0:
        return groups
    trimmed: List[List[pd.DataFrame]] = []
    for group in groups:
        trimmed.append([g.iloc[:min_len].reset_index(drop=True).copy() for g in group])
    return trimmed


def rotate_group_about_z(dfs: List[pd.DataFrame], theta_deg: float) -> List[pd.DataFrame]:
    """Rotate measured x/y/z columns about Z by theta_deg. Truth (bx/by/bz) unchanged.

    Rotation is applied after alignment to avoid altering alignment indices.
    Recomputes 'mag' from rotated x/y/z.
    """
    if not dfs:
        return dfs
    c = float(np.cos(np.deg2rad(theta_deg)))
    s = float(np.sin(np.deg2rad(theta_deg)))
    Rz = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=float)
    out_list: List[pd.DataFrame] = []
    for df in dfs:
        out = df.copy()
        if all(col in out.columns for col in ("x", "y", "z")):
            xyz = out[["x", "y", "z"]].to_numpy(dtype=float)
            rot = xyz @ Rz.T
            out[["x", "y", "z"]] = rot
            out["mag"] = np.sqrt(np.sum(rot ** 2, axis=1))
        out_list.append(out)
    return out_list


def compute_group_stats(dfs: List[pd.DataFrame], cols: List[str]) -> Dict[str, pd.DataFrame]:
    stats: Dict[str, pd.DataFrame] = {}
    for col in cols:
        arr = np.stack([df[col].to_numpy() for df in dfs], axis=0)
        mean = np.nanmean(arr, axis=0)
        std = np.nanstd(arr, axis=0)
        stats[col] = pd.DataFrame({"mean": mean, "std": std})
    return stats


def compute_group_stats_aligned(
    dfs: List[pd.DataFrame],
    cols: List[str],
    x_col: str = "frame_aligned",
) -> Tuple[Dict[str, pd.DataFrame], np.ndarray]:
    """Compute mean/std over a common aligned grid defined by x_col.

    Returns (stats_dict, grid_x). stats_dict[col] has columns 'mean' and 'std' and is indexed by grid_x.
    """
    if not dfs:
        return {}, np.array([])
    # Determine integer grid from min to max across all runs
    min_x = None
    max_x = None
    for df in dfs:
        if x_col not in df.columns:
            continue
        x = df[x_col].to_numpy()
        if x.size == 0:
            continue
        cur_min = int(np.nanmin(x))
        cur_max = int(np.nanmax(x))
        min_x = cur_min if min_x is None else min(min_x, cur_min)
        max_x = cur_max if max_x is None else max(max_x, cur_max)
    if min_x is None or max_x is None:
        return {}, np.array([])
    grid_x = np.arange(min_x, max_x + 1)

    stats: Dict[str, pd.DataFrame] = {}
    for col in cols:
        rows: List[np.ndarray] = []
        for df in dfs:
            if x_col not in df.columns or col not in df.columns:
                rows.append(np.full_like(grid_x, np.nan, dtype=float))
                continue
            series = df[[x_col, col]].dropna()
            s = series.set_index(x_col)[col]
            y = s.reindex(grid_x).to_numpy()
            rows.append(y.astype(float))
        arr = np.stack(rows, axis=0)
        mean = np.nanmean(arr, axis=0)
        std = np.nanstd(arr, axis=0)
        stats[col] = pd.DataFrame({"mean": mean, "std": std}, index=grid_x)
    return stats, grid_x


def _ensure_frame_aligned_column(dfs: List[pd.DataFrame]) -> None:
    """Ensure each DataFrame has a 'frame_aligned' integer axis."""
    for df in dfs:
        if "frame_aligned" not in df.columns:
            df["frame_aligned"] = np.arange(len(df), dtype=int)


def align_sets_by_first_x_peak(
    mounted: List[pd.DataFrame],
    reseated: List[pd.DataFrame],
    params: Params,
) -> Tuple[List[pd.DataFrame], List[pd.DataFrame]]:
    """Align reseated set to mounted set by the first local 'x' peak over threshold.

    - Uses group mean over an aligned grid.
    - Threshold uses Params.local_peak_threshold (default 1000.0).
    - If no threshold-crossing window is found, falls back to global |x| peak.
    """
    if not mounted or not reseated:
        return mounted, reseated

    _ensure_frame_aligned_column(mounted)
    _ensure_frame_aligned_column(reseated)

    m_stats, m_grid = compute_group_stats_aligned(mounted, ["x"], x_col="frame_aligned")
    r_stats, r_grid = compute_group_stats_aligned(reseated, ["x"], x_col="frame_aligned")
    if not m_stats or not r_stats:
        return mounted, reseated

    m_series = m_stats["x"]["mean"].to_numpy()
    r_series = r_stats["x"]["mean"].to_numpy()

    m_idx = find_local_peak_window_index(m_series, params.local_peak_threshold)
    if m_idx is None:
        m_idx = int(np.nanargmax(np.abs(m_series))) if m_series.size else 0
    r_idx = find_local_peak_window_index(r_series, params.local_peak_threshold)
    if r_idx is None:
        r_idx = int(np.nanargmax(np.abs(r_series))) if r_series.size else 0

    # Map to aligned frame coordinates
    m_frame = int(m_grid[m_idx]) if m_grid.size and 0 <= m_idx < m_grid.size else 0
    r_frame = int(r_grid[r_idx]) if r_grid.size and 0 <= r_idx < r_grid.size else 0

    delta = r_frame - m_frame
    if delta == 0:
        return mounted, reseated

    # Shift reseated runs so the selected peak aligns with mounted's
    out_reseated: List[pd.DataFrame] = []
    for df in reseated:
        out = df.copy()
        if "frame_aligned" in out.columns:
            out["frame_aligned"] = out["frame_aligned"].astype(int) - int(delta)
        out_reseated.append(out)
    return mounted, out_reseated


def compute_metrics(dfs: List[pd.DataFrame], mean_ref, cols: List[str]) -> pd.DataFrame:
    records = []
    for df in dfs:
        name = df.get("source_path", "run")
        for col in cols:
            y = df[col].to_numpy()
            # Resolve mean reference for this column
            mu_series = None
            if isinstance(mean_ref, dict) and col in mean_ref:
                mu_series = mean_ref[col]
            elif isinstance(mean_ref, pd.DataFrame) and col in mean_ref.columns:
                mu_series = mean_ref[col]
            else:
                mu_series = pd.Series(np.full_like(y, np.nan, dtype=float))

            # If mean is a Series with an aligned x-axis, reindex to this run's aligned frames
            if isinstance(mu_series, pd.Series) and "frame_aligned" in df.columns and mu_series.index.dtype.kind in ("i", "u", "f"):
                mu = mu_series.reindex(df["frame_aligned"]).to_numpy()
            else:
                mu = np.asarray(mu_series)

            # Safely handle different lengths by truncating to the overlap
            n = min(len(y), len(mu))
            if n <= 0:
                rmse = float("nan")
                r = float("nan")
                peak_idx = int(np.nan)
                peak_amp = float("nan")
                records.append({
                    "run": name,
                    "axis": col,
                    "rmse": rmse,
                    "pearson_r": r,
                    "peak_amp": peak_amp,
                    "peak_idx": peak_idx,
                })
                continue
            y_use = y[:n]
            mu_use = mu[:n]

            rmse = float(np.sqrt(np.nanmean((y_use - mu_use) ** 2)))
            # Pearson r
            if np.std(y_use) == 0 or np.std(mu_use) == 0:
                r = np.nan
            else:
                r = float(np.corrcoef(y_use, mu_use)[0, 1])
            # Peak and latency
            peak_idx = int(np.nanargmax(np.abs(y_use)))
            peak_amp = float(y_use[peak_idx])
            records.append(
                {
                    "run": name,
                    "axis": col,
                    "rmse": rmse,
                    "pearson_r": r,
                    "peak_amp": peak_amp,
                    "peak_idx": peak_idx,
                }
            )
    return pd.DataFrame.from_records(records)


def summarize_set(metrics_df: pd.DataFrame, stats: Dict[str, pd.DataFrame], axes: List[str]) -> pd.DataFrame:
    rows = []
    for axis in axes:
        m_axis = metrics_df.loc[metrics_df["axis"] == axis]
        rmse_mean = float(np.nanmean(m_axis["rmse"])) if not m_axis.empty else float("nan")
        rmse_std = float(np.nanstd(m_axis["rmse"])) if not m_axis.empty else float("nan")
        r_mean = float(np.nanmean(m_axis["pearson_r"])) if not m_axis.empty else float("nan")
        peak_amp_mean = float(np.nanmean(m_axis["peak_amp"])) if not m_axis.empty else float("nan")
        peak_amp_std = float(np.nanstd(m_axis["peak_amp"])) if not m_axis.empty else float("nan")
        peak_idx_mean = float(np.nanmean(m_axis["peak_idx"])) if not m_axis.empty else float("nan")
        peak_idx_std = float(np.nanstd(m_axis["peak_idx"])) if not m_axis.empty else float("nan")

        mean_band_std = float(np.nanmean(stats[axis]["std"])) if axis in stats else float("nan")

        rows.append({
            "axis": axis,
            "rmse_mean": rmse_mean,
            "rmse_std": rmse_std,
            "pearson_r_mean": r_mean,
            "peak_amp_mean": peak_amp_mean,
            "peak_amp_std": peak_amp_std,
            "peak_idx_mean": peak_idx_mean,
            "peak_idx_std": peak_idx_std,
            "mean_std_over_time": mean_band_std,
            "num_runs": int(len(m_axis))
        })
    return pd.DataFrame(rows)


def summarize_between_sets(m_stats: Dict[str, pd.DataFrame], r_stats: Dict[str, pd.DataFrame], axes: List[str]) -> pd.DataFrame:
    rows = []
    for axis in axes:
        if axis not in m_stats or axis not in r_stats:
            rows.append({"axis": axis, "rmse_between_means": float("nan"), "pearson_r_between_means": float("nan"), "delta_peak_idx": float("nan"), "delta_peak_amp": float("nan")})
            continue
        a = m_stats[axis]["mean"].to_numpy()
        b = r_stats[axis]["mean"].to_numpy()
        n = min(len(a), len(b))
        if n <= 0:
            rows.append({"axis": axis, "rmse_between_means": float("nan"), "pearson_r_between_means": float("nan"), "delta_peak_idx": float("nan"), "delta_peak_amp": float("nan")})
            continue
        a = a[:n]
        b = b[:n]
        rmse_means = float(np.sqrt(np.nanmean((a - b) ** 2)))
        r_means = float(np.corrcoef(a, b)[0, 1]) if (np.std(a) != 0 and np.std(b) != 0) else float("nan")
        a_peak_idx = int(np.nanargmax(np.abs(a))) if np.size(a) else 0
        b_peak_idx = int(np.nanargmax(np.abs(b))) if np.size(b) else 0
        rows.append({
            "axis": axis,
            "rmse_between_means": rmse_means,
            "pearson_r_between_means": r_means,
            "delta_peak_idx": float(b_peak_idx - a_peak_idx),
            "delta_peak_amp": float(b[b_peak_idx] - a[a_peak_idx]) if n > 0 else float("nan")
        })
    return pd.DataFrame(rows)

def plot_overlaid(
    mounted: List[pd.DataFrame],
    reseated: List[pd.DataFrame],
    paths: Paths,
    params: Params,
) -> None:
    sns.set_style("whitegrid")
    cols = ["x", "y", "z", "mag"]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes_map = {"x": (0, 0), "y": (0, 1), "z": (1, 0), "mag": (1, 1)}

    # Collect artists for interactive toggling
    mounted_lines: list = []
    reseated_lines: list = []
    mounted_mean_lines: list = []
    reseated_mean_lines: list = []
    mounted_bands: list = []
    reseated_bands: list = []
    truth_mounted_lines: list = []
    truth_reseated_lines: list = []
    truth_mounted_mean_lines: list = []
    truth_reseated_mean_lines: list = []
    truth_mounted_bands: list = []
    truth_reseated_bands: list = []

    for col in cols:
        r, c = axes_map[col]
        ax = axes[r][c]
        for df in mounted:
            x_vals = df["frame_aligned"] if (params.alignment_method in ("peak", "local_peak") and "frame_aligned" in df.columns) else df.index
            line, = ax.plot(x_vals, df[col], color="#1f77b4", alpha=0.5, linewidth=1.0)
            mounted_lines.append(line)
        for df in reseated:
            x_vals = df["frame_aligned"] if (params.alignment_method in ("peak", "local_peak") and "frame_aligned" in df.columns) else df.index
            line, = ax.plot(x_vals, df[col], color="#d62728", alpha=0.5, linewidth=1.0)
            reseated_lines.append(line)
        ax.set_title(f"{col.upper()} vs Frame")
        ax.set_xlabel("Frame")
        ax.set_ylabel(col.upper())

        # If truth data exists, plot it on matching subplot
        if all(col_name in mounted[0].columns for col_name in ("bx", "by", "bz")):
            bcol = {"x": "bx", "y": "by", "z": "bz", "mag": "bmag"}[col]
            # Some runs may lack truth; guard per-df
            for df in mounted:
                if bcol in df.columns:
                    x_vals = df["frame_aligned"] if (params.alignment_method in ("peak", "local_peak") and "frame_aligned" in df.columns) else df.index
                    tl, = ax.plot(x_vals, df[bcol], color="#222222", alpha=0.5, linewidth=1.0)
                    truth_mounted_lines.append(tl)
            for df in reseated:
                if bcol in df.columns:
                    x_vals = df["frame_aligned"] if (params.alignment_method in ("peak", "local_peak") and "frame_aligned" in df.columns) else df.index
                    tl, = ax.plot(x_vals, df[bcol], color="#666666", alpha=0.5, linewidth=1.0)
                    truth_reseated_lines.append(tl)

    # Optional mean bands
    if params.plot_mean_band and mounted and reseated:
        if (params.alignment_method in ("peak", "local_peak")) and all(("frame_aligned" in df.columns) for df in mounted) and all(("frame_aligned" in df.columns) for df in reseated):
            m_stats, _ = compute_group_stats_aligned(mounted, cols, x_col="frame_aligned")
            r_stats, _ = compute_group_stats_aligned(reseated, cols, x_col="frame_aligned")
        else:
            m_stats = compute_group_stats(mounted, cols)
            r_stats = compute_group_stats(reseated, cols)
        # Truth group stats if available (compute per set)
        truth_available_m = bool(mounted) and all(c in mounted[0].columns for c in ("bx", "by", "bz"))
        truth_available_r = bool(reseated) and all(c in reseated[0].columns for c in ("bx", "by", "bz"))
        if truth_available_m or truth_available_r:
            t_cols_map = {"x": "bx", "y": "by", "z": "bz", "mag": "bmag"}
            truth_for_stats_m: List[pd.DataFrame] = []
            truth_for_stats_r: List[pd.DataFrame] = []
            if truth_available_m:
                for df in mounted:
                    if all(v in df.columns for v in t_cols_map.values()):
                        tdf = pd.DataFrame({k: df[v] for k, v in t_cols_map.items()})
                        if params.alignment_method in ("peak", "local_peak") and "frame_aligned" in df.columns:
                            tdf["frame_aligned"] = df["frame_aligned"].to_numpy()
                        truth_for_stats_m.append(tdf)
            if truth_available_r:
                for df in reseated:
                    if all(v in df.columns for v in t_cols_map.values()):
                        tdf = pd.DataFrame({k: df[v] for k, v in t_cols_map.items()})
                        if params.alignment_method in ("peak", "local_peak") and "frame_aligned" in df.columns:
                            tdf["frame_aligned"] = df["frame_aligned"].to_numpy()
                        truth_for_stats_r.append(tdf)

            if truth_for_stats_m:
                if params.alignment_method in ("peak", "local_peak"):
                    for i in range(len(truth_for_stats_m)):
                        if "frame_aligned" not in truth_for_stats_m[i].columns:
                            truth_for_stats_m[i]["frame_aligned"] = np.arange(len(truth_for_stats_m[i]))
                    t_m_stats, _ = compute_group_stats_aligned(truth_for_stats_m, cols, x_col="frame_aligned")
                else:
                    t_m_stats = compute_group_stats(truth_for_stats_m, cols)
            if truth_for_stats_r:
                if params.alignment_method in ("peak", "local_peak"):
                    for i in range(len(truth_for_stats_r)):
                        if "frame_aligned" not in truth_for_stats_r[i].columns:
                            truth_for_stats_r[i]["frame_aligned"] = np.arange(len(truth_for_stats_r[i]))
                    t_r_stats, _ = compute_group_stats_aligned(truth_for_stats_r, cols, x_col="frame_aligned")
                else:
                    t_r_stats = compute_group_stats(truth_for_stats_r, cols)
        for col in cols:
            r, c = axes_map[col]
            ax = axes[r][c]
            idx = m_stats[col].index.to_numpy() if hasattr(m_stats[col], 'index') else np.arange(len(m_stats[col]["mean"]))
            m_mean_line, = ax.plot(idx, m_stats[col]["mean"], color="#1f77b4", linewidth=2.0, label="Mounted mean")
            mounted_mean_lines.append(m_mean_line)
            m_band = ax.fill_between(
                idx,
                m_stats[col]["mean"] - m_stats[col]["std"],
                m_stats[col]["mean"] + m_stats[col]["std"],
                color="#1f77b4",
                alpha=0.15,
                label="Mounted ±1 SD",
            )
            mounted_bands.append(m_band)
            idx = r_stats[col].index.to_numpy() if hasattr(r_stats[col], 'index') else np.arange(len(r_stats[col]["mean"]))
            r_mean_line, = ax.plot(idx, r_stats[col]["mean"], color="#d62728", linewidth=2.0, label="Reseated mean")
            reseated_mean_lines.append(r_mean_line)
            r_band = ax.fill_between(
                idx,
                r_stats[col]["mean"] - r_stats[col]["std"],
                r_stats[col]["mean"] + r_stats[col]["std"],
                color="#d62728",
                alpha=0.15,
                label="Reseated ±1 SD",
            )
            reseated_bands.append(r_band)
            ax.legend(loc="upper right", fontsize=8)

            # Truth mean ± SD overlays per set if available
            if truth_available_m and 't_m_stats' in locals() and col in t_m_stats:
                idx = t_m_stats[col].index.to_numpy() if hasattr(t_m_stats[col], 'index') else np.arange(len(t_m_stats[col]["mean"]))
                t_m_mean_line, = ax.plot(idx, t_m_stats[col]["mean"], color="#222222", linewidth=2.0, label="Mounted truth mean")
                truth_mounted_mean_lines.append(t_m_mean_line)
                t_m_band = ax.fill_between(
                    idx,
                    t_m_stats[col]["mean"] - t_m_stats[col]["std"],
                    t_m_stats[col]["mean"] + t_m_stats[col]["std"],
                    color="#222222",
                    alpha=0.10,
                    label="Mounted truth ±1 SD",
                )
                truth_mounted_bands.append(t_m_band)
            if truth_available_r and 't_r_stats' in locals() and col in t_r_stats:
                idx = t_r_stats[col].index.to_numpy() if hasattr(t_r_stats[col], 'index') else np.arange(len(t_r_stats[col]["mean"]))
                t_r_mean_line, = ax.plot(idx, t_r_stats[col]["mean"], color="#666666", linewidth=2.0, label="Reseated truth mean")
                truth_reseated_mean_lines.append(t_r_mean_line)
                t_r_band = ax.fill_between(
                    idx,
                    t_r_stats[col]["mean"] - t_r_stats[col]["std"],
                    t_r_stats[col]["mean"] + t_r_stats[col]["std"],
                    color="#666666",
                    alpha=0.10,
                    label="Reseated truth ±1 SD",
                )
                truth_reseated_bands.append(t_r_band)

    # Add interactive toggles for runs and means
    fig.subplots_adjust(right=0.85)
    rax = fig.add_axes([0.87, 0.35, 0.12, 0.3])  # [left, bottom, width, height]
    labels = []
    artists_map: dict = {}
    actives: list = []

    if mounted_lines:
        labels.append("Mounted runs")
        artists_map["Mounted runs"] = mounted_lines
        actives.append(True)
    if reseated_lines:
        labels.append("Reseated runs")
        artists_map["Reseated runs"] = reseated_lines
        actives.append(True)
    if mounted_mean_lines or mounted_bands:
        labels.append("Mounted mean±SD")
        artists_map["Mounted mean±SD"] = mounted_mean_lines + mounted_bands
        actives.append(True)
    if reseated_mean_lines or reseated_bands:
        labels.append("Reseated mean±SD")
        artists_map["Reseated mean±SD"] = reseated_mean_lines + reseated_bands
        actives.append(True)
    if truth_mounted_lines:
        labels.append("Mounted truth runs")
        artists_map["Mounted truth runs"] = truth_mounted_lines
        actives.append(True)
    if truth_reseated_lines:
        labels.append("Reseated truth runs")
        artists_map["Reseated truth runs"] = truth_reseated_lines
        actives.append(True)
    if truth_mounted_mean_lines or truth_mounted_bands:
        labels.append("Mounted truth mean±SD")
        artists_map["Mounted truth mean±SD"] = truth_mounted_mean_lines + truth_mounted_bands
        actives.append(True)
    if truth_reseated_mean_lines or truth_reseated_bands:
        labels.append("Reseated truth mean±SD")
        artists_map["Reseated truth mean±SD"] = truth_reseated_mean_lines + truth_reseated_bands
        actives.append(True)

    if labels:
        check = CheckButtons(rax, labels=labels, actives=actives)

        def on_toggle(label: str) -> None:
            # Set visibility based on checkbox state
            try:
                idx = labels.index(label)
            except ValueError:
                return
            visible = check.get_status()[idx]
            for artist in artists_map.get(label, []):
                artist.set_visible(visible)
            for a in axes.ravel():
                a.figure.canvas.draw_idle()

        check.on_clicked(on_toggle)

    fig.suptitle("Load Cell Sensitivity: Mounted (blue) vs Reseated (red) vs Truth (black/gray)")
    # Show interactively instead of saving for now
    plt.show()


def main() -> None:
    paths = Paths()
    params = Params()
    ensure_dirs(paths)

    # Load data
    mounted_raw = load_group(paths.mounted_glob, params.axis_mapping)
    reseated_raw = load_group(paths.reseated_glob, params.axis_mapping)

    # Preprocess
    mounted_prep = [preprocess_signal(df, params) for df in mounted_raw]
    reseated_prep = [preprocess_signal(df, params) for df in reseated_raw]

    # Align
    if params.alignment_method == "peak":
        # Align peaks separately per set using mag
        mounted_aligned = align_group_by_peak(mounted_prep, params, signal="mag")
        reseated_aligned = align_group_by_peak(reseated_prep, params, signal="mag")
        # No trimming; preserve full window with aligned x-axis
        mounted_trimmed, reseated_trimmed = mounted_aligned, reseated_aligned
    elif params.alignment_method == "local_peak":
        mounted_aligned = align_group_by_local_peak(mounted_prep, params)
        reseated_aligned = align_group_by_local_peak(reseated_prep, params)
        mounted_trimmed, reseated_trimmed = mounted_aligned, reseated_aligned
    else:
        mounted_aligned = [align_run(df, params) for df in mounted_prep]
        reseated_aligned = [align_run(df, params) for df in reseated_prep]
        # Trim to common length for fair overlay and metrics
        (mounted_trimmed, reseated_trimmed) = trim_to_common_length([mounted_aligned, reseated_aligned])

    # Rotate measured axes after alignment (preserve alignment indices)
    mounted_rot = rotate_group_about_z(mounted_trimmed, params.rotation_deg_z)
    reseated_rot = rotate_group_about_z(reseated_trimmed, params.rotation_deg_z)

    # Inter-set alignment by first local 'x' peak over threshold
    mounted_adj, reseated_adj = align_sets_by_first_x_peak(mounted_rot, reseated_rot, params)

    # Plot
    plot_overlaid(mounted_adj, reseated_adj, paths, params)

    # Metrics
    cols = ["x", "y", "z", "mag"]
    if params.alignment_method in ("peak", "local_peak"):
        # Compute metrics on aligned grid per set
        m_stats, _ = compute_group_stats_aligned(mounted_adj, cols, x_col="frame_aligned")
        r_stats, _ = compute_group_stats_aligned(reseated_adj, cols, x_col="frame_aligned")
        # Convert mean frames to arrays aligned to their own grid
        m_metrics = compute_metrics(mounted_adj, {k: v["mean"] for k, v in m_stats.items()}, cols)  # type: ignore
        r_metrics = compute_metrics(reseated_adj, {k: v["mean"] for k, v in r_stats.items()}, cols)  # type: ignore
    else:
        m_stats = compute_group_stats(mounted_adj, cols)
        r_stats = compute_group_stats(reseated_adj, cols)
        m_metrics = compute_metrics(mounted_adj, {k: v["mean"] for k, v in m_stats.items()}, cols)  # type: ignore
        r_metrics = compute_metrics(reseated_adj, {k: v["mean"] for k, v in r_stats.items()}, cols)  # type: ignore

    # Save compact summaries (per-set and between-set)
    mounted_summary = summarize_set(m_metrics, m_stats, cols)
    reseated_summary = summarize_set(r_metrics, r_stats, cols)
    between_summary = summarize_between_sets(m_stats, r_stats, cols)
    mounted_summary.to_csv(os.path.join(paths.metrics_dir, "mounted_summary.csv"), index=False)
    reseated_summary.to_csv(os.path.join(paths.metrics_dir, "reseated_summary.csv"), index=False)
    between_summary.to_csv(os.path.join(paths.metrics_dir, "between_sets_summary.csv"), index=False)

    print("Saved summaries:")
    print(f"- {os.path.join(paths.metrics_dir, 'mounted_summary.csv')}")
    print(f"- {os.path.join(paths.metrics_dir, 'reseated_summary.csv')}")
    print(f"- {os.path.join(paths.metrics_dir, 'between_sets_summary.csv')}")


if __name__ == "__main__":
    main()


