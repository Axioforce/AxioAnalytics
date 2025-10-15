import os
import glob
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt
import matplotlib.pyplot as plt
import seaborn as sns
import mplcursors
from matplotlib.widgets import CheckButtons
from pathlib import Path


@dataclass
class Paths:
    # Resolve paths relative to the project folder (one level above this script)
    _project_root: Path = Path(__file__).resolve().parent.parent
    mounted_glob: str = str(_project_root / "Load_Cell_Spiral_test" / "H6b.90" / "10.08.2025" / "mounted_runs" / "*.csv")
    reseated_glob: str = str(_project_root / "Load_Cell_Spiral_test" / "H6b.90" / "10.08.2025" / "stationary_runs" / "*.csv")


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
    subtract_baseline: bool = False  # if True, subtract per-run median baseline
    # Alignment-only smoothing (minimal) applied internally before peak-finding
    align_smoothing_enabled: bool = True
    align_smoothing_window: int = 17  # frames; should be odd

    def __post_init__(self) -> None:
        if self.axis_mapping is None:
            # Default to summed axes. Change to e.g. '1-inner-x' if you need a specific cell.
            self.axis_mapping = {"x": "sum-x", "y": "sum-y", "z": "sum-z"}


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
        # Compute magnitude for SUM (default mapping)
        tidy.rename(columns={axis_mapping["x"]: "x", axis_mapping["y"]: "y", axis_mapping["z"]: "z"}, inplace=True)
        tidy["mag"] = np.sqrt(tidy["x"] ** 2 + tidy["y"] ** 2 + tidy["z"] ** 2)

        # Compute INNER and OUTER aggregates from raw columns if available
        try:
            inner_x_cols = [f"{i}-inner-x" for i in (0, 1, 2, 3)]
            inner_y_cols = [f"{i}-inner-y" for i in (0, 1, 2, 3)]
            inner_z_cols = [f"{i}-inner-z" for i in (0, 1, 2, 3)]
            if all(c in df.columns for c in inner_x_cols + inner_y_cols + inner_z_cols):
                tidy["x_inner"] = df[inner_x_cols].sum(axis=1)
                tidy["y_inner"] = df[inner_y_cols].sum(axis=1)
                tidy["z_inner"] = df[inner_z_cols].sum(axis=1)
                tidy["mag_inner"] = np.sqrt(tidy["x_inner"] ** 2 + tidy["y_inner"] ** 2 + tidy["z_inner"] ** 2)
        except Exception:
            pass
        try:
            outer_x_cols = [f"{i}-outer-x" for i in (0, 1, 2, 3)]
            outer_y_cols = [f"{i}-outer-y" for i in (0, 1, 2, 3)]
            outer_z_cols = [f"{i}-outer-z" for i in (0, 1, 2, 3)]
            if all(c in df.columns for c in outer_x_cols + outer_y_cols + outer_z_cols):
                tidy["x_outer"] = df[outer_x_cols].sum(axis=1)
                tidy["y_outer"] = df[outer_y_cols].sum(axis=1)
                tidy["z_outer"] = df[outer_z_cols].sum(axis=1)
                tidy["mag_outer"] = np.sqrt(tidy["x_outer"] ** 2 + tidy["y_outer"] ** 2 + tidy["z_outer"] ** 2)
        except Exception:
            pass
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


def smooth_for_alignment(y: np.ndarray, window: int) -> np.ndarray:
    """Minimal moving-average smoothing for alignment-only peak detection.

    Does not mutate original data; used only to improve robustness of peak-finding.
    """
    if y.size == 0:
        return y
    w = int(window) if window is not None else 0
    if w <= 1:
        return y
    if (w % 2) == 0:
        w += 1  # prefer odd window for symmetry
    kernel = np.ones(w, dtype=float) / float(w)
    return np.convolve(y.astype(float), kernel, mode="same")

def preprocess_signal(df: pd.DataFrame, params: Params) -> pd.DataFrame:
    out = df.copy()
    # Optional baseline subtraction per run
    if params.subtract_baseline:
        # Baseline: using first N seconds if time exists, else first N samples as approx
        if "time" in out.columns:
            t0 = out["time"].iloc[0]
            baseline_mask = out["time"] <= (t0 + params.baseline_seconds * 1000.0)
        else:
            n = max(1, int(0.5 * len(out)))  # fallback: 50% of start; conservative
            baseline_mask = pd.Series([True] * n + [False] * (len(out) - n))

        candidate_cols = (
            "x", "y", "z", "mag",
            "x_inner", "y_inner", "z_inner", "mag_inner",
            "x_outer", "y_outer", "z_outer", "mag_outer",
            "bx", "by", "bz", "bmag",
        )
        for col in candidate_cols:
            if col in out.columns:
                baseline = out.loc[baseline_mask, col].median()
                out[col] = out[col] - baseline

    if params.apply_filter:
        fs_hz = estimate_sampling_rate_hz(out)
        if not np.isnan(fs_hz) and fs_hz > 0 and params.filter_cutoff_hz < 0.5 * fs_hz:
            b, a = butter_lowpass(params.filter_cutoff_hz, fs_hz, params.filter_order)
            filter_cols = (
                "x", "y", "z", "mag",
                "x_inner", "y_inner", "z_inner", "mag_inner",
                "x_outer", "y_outer", "z_outer", "mag_outer",
            )
            for col in filter_cols:
                if col in out.columns:
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
        x_for_peak = smooth_for_alignment(x, params.align_smoothing_window) if params.align_smoothing_enabled else x
        idx = find_local_peak_window_index(x_for_peak, thr)
        if idx is None:
            # fallback to global |x| peak
            idx = int(np.nanargmax(np.abs(x_for_peak)))
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
        # Sum variant
        if all(col in out.columns for col in ("x", "y", "z")):
            xyz = out[["x", "y", "z"]].to_numpy(dtype=float)
            rot = xyz @ Rz.T
            out[["x", "y", "z"]] = rot
            out["mag"] = np.sqrt(np.sum(rot ** 2, axis=1))
        # Inner variant
        if all(col in out.columns for col in ("x_inner", "y_inner", "z_inner")):
            xyz_i = out[["x_inner", "y_inner", "z_inner"]].to_numpy(dtype=float)
            rot_i = xyz_i @ Rz.T
            out[["x_inner", "y_inner", "z_inner"]] = rot_i
            out["mag_inner"] = np.sqrt(np.sum(rot_i ** 2, axis=1))
        # Outer variant
        if all(col in out.columns for col in ("x_outer", "y_outer", "z_outer")):
            xyz_o = out[["x_outer", "y_outer", "z_outer"]].to_numpy(dtype=float)
            rot_o = xyz_o @ Rz.T
            out[["x_outer", "y_outer", "z_outer"]] = rot_o
            out["mag_outer"] = np.sqrt(np.sum(rot_o ** 2, axis=1))
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
    if params.align_smoothing_enabled:
        m_series = smooth_for_alignment(m_series, params.align_smoothing_window)
        r_series = smooth_for_alignment(r_series, params.align_smoothing_window)

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


def plot_overlaid(
    mounted: List[pd.DataFrame],
    reseated: List[pd.DataFrame],
    paths: Paths,
    params: Params,
) -> None:
    sns.set_style("whitegrid")
    cols = ["x", "y", "z", "mag"]
    fig, axes = plt.subplots(2, 2, figsize=(16, 9))
    axes_map = {"x": (0, 0), "y": (0, 1), "z": (1, 0), "mag": (1, 1)}

    # Collect artists for interactive toggling
    mounted_lines: list = []  # sum variant runs
    reseated_lines: list = []
    mounted_mean_lines: list = []  # sum variant means
    reseated_mean_lines: list = []
    mounted_bands: list = []  # sum variant bands
    reseated_bands: list = []

    inner_mounted_lines: list = []
    inner_reseated_lines: list = []
    inner_mounted_mean_lines: list = []
    inner_reseated_mean_lines: list = []
    inner_mounted_bands: list = []
    inner_reseated_bands: list = []

    outer_mounted_lines: list = []
    outer_reseated_lines: list = []
    outer_mounted_mean_lines: list = []
    outer_reseated_mean_lines: list = []
    outer_mounted_bands: list = []
    outer_reseated_bands: list = []
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
            line, = ax.plot(x_vals, df[col], color="#1f77b4", alpha=0.5, linewidth=1.0, linestyle="-")
            if "source_path" in df.columns:
                try:
                    line._source_path = str(df["source_path"].iloc[0])
                except Exception:
                    pass
            mounted_lines.append(line)
        # Inner variant
        inner_col = f"{col}_inner" if col != "mag" else "mag_inner"
        if inner_col in mounted[0].columns:
            for df in mounted:
                if inner_col not in df.columns:
                    continue
                x_vals = df["frame_aligned"] if (params.alignment_method in ("peak", "local_peak") and "frame_aligned" in df.columns) else df.index
                line, = ax.plot(x_vals, df[inner_col], color="#1f77b4", alpha=0.5, linewidth=1.0, linestyle="--")
                if "source_path" in df.columns:
                    try:
                        line._source_path = str(df["source_path"].iloc[0])
                    except Exception:
                        pass
                inner_mounted_lines.append(line)
        # Outer variant
        outer_col = f"{col}_outer" if col != "mag" else "mag_outer"
        if outer_col in mounted[0].columns:
            for df in mounted:
                if outer_col not in df.columns:
                    continue
                x_vals = df["frame_aligned"] if (params.alignment_method in ("peak", "local_peak") and "frame_aligned" in df.columns) else df.index
                line, = ax.plot(x_vals, df[outer_col], color="#1f77b4", alpha=0.5, linewidth=1.0, linestyle=":")
                if "source_path" in df.columns:
                    try:
                        line._source_path = str(df["source_path"].iloc[0])
                    except Exception:
                        pass
                outer_mounted_lines.append(line)

        for df in reseated:
            x_vals = df["frame_aligned"] if (params.alignment_method in ("peak", "local_peak") and "frame_aligned" in df.columns) else df.index
            line, = ax.plot(x_vals, df[col], color="#d62728", alpha=0.5, linewidth=1.0, linestyle="-")
            if "source_path" in df.columns:
                try:
                    line._source_path = str(df["source_path"].iloc[0])
                except Exception:
                    pass
            reseated_lines.append(line)
        # Inner variant
        if inner_col in reseated[0].columns:
            for df in reseated:
                if inner_col not in df.columns:
                    continue
                x_vals = df["frame_aligned"] if (params.alignment_method in ("peak", "local_peak") and "frame_aligned" in df.columns) else df.index
                line, = ax.plot(x_vals, df[inner_col], color="#d62728", alpha=0.5, linewidth=1.0, linestyle="--")
                if "source_path" in df.columns:
                    try:
                        line._source_path = str(df["source_path"].iloc[0])
                    except Exception:
                        pass
                inner_reseated_lines.append(line)
        # Outer variant
        if outer_col in reseated[0].columns:
            for df in reseated:
                if outer_col not in df.columns:
                    continue
                x_vals = df["frame_aligned"] if (params.alignment_method in ("peak", "local_peak") and "frame_aligned" in df.columns) else df.index
                line, = ax.plot(x_vals, df[outer_col], color="#d62728", alpha=0.5, linewidth=1.0, linestyle=":")
                if "source_path" in df.columns:
                    try:
                        line._source_path = str(df["source_path"].iloc[0])
                    except Exception:
                        pass
                outer_reseated_lines.append(line)
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
                    if "source_path" in df.columns:
                        try:
                            tl._source_path = str(df["source_path"].iloc[0])
                        except Exception:
                            pass
                    truth_mounted_lines.append(tl)
            for df in reseated:
                if bcol in df.columns:
                    x_vals = df["frame_aligned"] if (params.alignment_method in ("peak", "local_peak") and "frame_aligned" in df.columns) else df.index
                    tl, = ax.plot(x_vals, df[bcol], color="#666666", alpha=0.5, linewidth=1.0)
                    if "source_path" in df.columns:
                        try:
                            tl._source_path = str(df["source_path"].iloc[0])
                        except Exception:
                            pass
                    truth_reseated_lines.append(tl)

    # Optional mean bands
    if params.plot_mean_band and mounted and reseated:
        # Stats for sum + inner + outer
        all_cols = [
            "x", "y", "z", "mag",
            "x_inner", "y_inner", "z_inner", "mag_inner",
            "x_outer", "y_outer", "z_outer", "mag_outer",
        ]
        if (params.alignment_method in ("peak", "local_peak")) and all(("frame_aligned" in df.columns) for df in mounted) and all(("frame_aligned" in df.columns) for df in reseated):
            m_stats, _ = compute_group_stats_aligned(mounted, all_cols, x_col="frame_aligned")
            r_stats, _ = compute_group_stats_aligned(reseated, all_cols, x_col="frame_aligned")
        else:
            m_stats = compute_group_stats(mounted, all_cols)
            r_stats = compute_group_stats(reseated, all_cols)
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
            # Inner means/bands if present
            inner_col = f"{col}_inner" if col != "mag" else "mag_inner"
            if inner_col in m_stats and inner_col in r_stats:
                idx_i = m_stats[inner_col].index.to_numpy() if hasattr(m_stats[inner_col], 'index') else np.arange(len(m_stats[inner_col]["mean"]))
                m_i_line, = ax.plot(idx_i, m_stats[inner_col]["mean"], color="#1f77b4", linewidth=2.0, linestyle="--")
                inner_mounted_mean_lines.append(m_i_line)
                m_i_band = ax.fill_between(idx_i, m_stats[inner_col]["mean"] - m_stats[inner_col]["std"], m_stats[inner_col]["mean"] + m_stats[inner_col]["std"], color="#1f77b4", alpha=0.12)
                inner_mounted_bands.append(m_i_band)
                idx_i = r_stats[inner_col].index.to_numpy() if hasattr(r_stats[inner_col], 'index') else np.arange(len(r_stats[inner_col]["mean"]))
                r_i_line, = ax.plot(idx_i, r_stats[inner_col]["mean"], color="#d62728", linewidth=2.0, linestyle="--")
                inner_reseated_mean_lines.append(r_i_line)
                r_i_band = ax.fill_between(idx_i, r_stats[inner_col]["mean"] - r_stats[inner_col]["std"], r_stats[inner_col]["mean"] + r_stats[inner_col]["std"], color="#d62728", alpha=0.12)
                inner_reseated_bands.append(r_i_band)
            # Outer means/bands if present
            outer_col = f"{col}_outer" if col != "mag" else "mag_outer"
            if outer_col in m_stats and outer_col in r_stats:
                idx_o = m_stats[outer_col].index.to_numpy() if hasattr(m_stats[outer_col], 'index') else np.arange(len(m_stats[outer_col]["mean"]))
                m_o_line, = ax.plot(idx_o, m_stats[outer_col]["mean"], color="#1f77b4", linewidth=2.0, linestyle=":")
                outer_mounted_mean_lines.append(m_o_line)
                m_o_band = ax.fill_between(idx_o, m_stats[outer_col]["mean"] - m_stats[outer_col]["std"], m_stats[outer_col]["mean"] + m_stats[outer_col]["std"], color="#1f77b4", alpha=0.10)
                outer_mounted_bands.append(m_o_band)
                idx_o = r_stats[outer_col].index.to_numpy() if hasattr(r_stats[outer_col], 'index') else np.arange(len(r_stats[outer_col]["mean"]))
                r_o_line, = ax.plot(idx_o, r_stats[outer_col]["mean"], color="#d62728", linewidth=2.0, linestyle=":")
                outer_reseated_mean_lines.append(r_o_line)
                r_o_band = ax.fill_between(idx_o, r_stats[outer_col]["mean"] - r_stats[outer_col]["std"], r_stats[outer_col]["mean"] + r_stats[outer_col]["std"], color="#d62728", alpha=0.10)
                outer_reseated_bands.append(r_o_band)

            ax.legend(loc="upper right", fontsize=8)

            # Truth mean ± SD overlays per set if available
            if 't_m_stats' in locals() and col in locals().get('t_m_stats', {}):
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
            if 't_r_stats' in locals() and col in locals().get('t_r_stats', {}):
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

    # Add interactive toggles for runs and means, split into sections
    fig.subplots_adjust(right=0.82, wspace=0.25, hspace=0.28)

    # Truth section (bottom)
    t_ax = fig.add_axes([0.86, 0.06, 0.12, 0.24])
    t_labels = []
    t_artists_map: dict = {}
    t_actives: list = []
    if truth_mounted_lines:
        t_labels.append("Mounted truth runs")
        t_artists_map["Mounted truth runs"] = truth_mounted_lines
        t_actives.append(True)
    if truth_mounted_mean_lines or truth_mounted_bands:
        t_labels.append("Mounted truth mean±SD")
        t_artists_map["Mounted truth mean±SD"] = truth_mounted_mean_lines + truth_mounted_bands
        t_actives.append(True)
    if truth_reseated_lines:
        t_labels.append("Reseated truth runs")
        t_artists_map["Reseated truth runs"] = truth_reseated_lines
        t_actives.append(True)
    if truth_reseated_mean_lines or truth_reseated_bands:
        t_labels.append("Reseated truth mean±SD")
        t_artists_map["Reseated truth mean±SD"] = truth_reseated_mean_lines + truth_reseated_bands
        t_actives.append(True)
    if t_labels:
        t_check = CheckButtons(t_ax, labels=t_labels, actives=t_actives)
        try:
            for txt in t_check.labels:
                txt.set_fontsize(10)
        except Exception:
            pass
        try:
            t_ax.set_title("Truth", fontsize=10, pad=2)
        except Exception:
            pass

        def on_toggle_t(label: str) -> None:
            try:
                idx = t_labels.index(label)
            except ValueError:
                return
            visible = t_check.get_status()[idx]
            for artist in t_artists_map.get(label, []):
                artist.set_visible(visible)
            for a in axes.ravel():
                a.figure.canvas.draw_idle()

        t_check.on_clicked(on_toggle_t)

    # Mounted section (below hover)
    m_ax = fig.add_axes([0.86, 0.60, 0.12, 0.24])
    m_labels = []
    m_artists_map: dict = {}
    m_actives: list = []
    if mounted_lines:
        m_labels.append("Sum lines")
        m_artists_map["Sum lines"] = mounted_lines
        m_actives.append(True)
    if mounted_mean_lines or mounted_bands:
        m_labels.append("Sum mean±SD")
        m_artists_map["Sum mean±SD"] = mounted_mean_lines + mounted_bands
        m_actives.append(True)
    if inner_mounted_lines:
        m_labels.append("Inner lines")
        m_artists_map["Inner lines"] = inner_mounted_lines
        m_actives.append(True)
    if inner_mounted_mean_lines or inner_mounted_bands:
        m_labels.append("Inner mean±SD")
        m_artists_map["Inner mean±SD"] = inner_mounted_mean_lines + inner_mounted_bands
        m_actives.append(True)
    if outer_mounted_lines:
        m_labels.append("Outer lines")
        m_artists_map["Outer lines"] = outer_mounted_lines
        m_actives.append(True)
    if outer_mounted_mean_lines or outer_mounted_bands:
        m_labels.append("Outer mean±SD")
        m_artists_map["Outer mean±SD"] = outer_mounted_mean_lines + outer_mounted_bands
        m_actives.append(True)
    if m_labels:
        m_check = CheckButtons(m_ax, labels=m_labels, actives=m_actives)
        try:
            for txt in m_check.labels:
                txt.set_fontsize(10)
        except Exception:
            pass
        try:
            m_ax.set_title("Mounted", fontsize=10, pad=2)
        except Exception:
            pass

        def on_toggle_m(label: str) -> None:
            try:
                idx = m_labels.index(label)
            except ValueError:
                return
            visible = m_check.get_status()[idx]
            for artist in m_artists_map.get(label, []):
                artist.set_visible(visible)
            for a in axes.ravel():
                a.figure.canvas.draw_idle()

        m_check.on_clicked(on_toggle_m)

    # Reseated section (between mounted and truth)
    r_ax = fig.add_axes([0.86, 0.33, 0.12, 0.24])
    r_labels = []
    r_artists_map: dict = {}
    r_actives: list = []
    if reseated_lines:
        r_labels.append("Sum lines")
        r_artists_map["Sum lines"] = reseated_lines
        r_actives.append(True)
    if reseated_mean_lines or reseated_bands:
        r_labels.append("Sum mean±SD")
        r_artists_map["Sum mean±SD"] = reseated_mean_lines + reseated_bands
        r_actives.append(True)
    if inner_reseated_lines:
        r_labels.append("Inner lines")
        r_artists_map["Inner lines"] = inner_reseated_lines
        r_actives.append(True)
    if inner_reseated_mean_lines or inner_reseated_bands:
        r_labels.append("Inner mean±SD")
        r_artists_map["Inner mean±SD"] = inner_reseated_mean_lines + inner_reseated_bands
        r_actives.append(True)
    if outer_reseated_lines:
        r_labels.append("Outer lines")
        r_artists_map["Outer lines"] = outer_reseated_lines
        r_actives.append(True)
    if outer_reseated_mean_lines or outer_reseated_bands:
        r_labels.append("Outer mean±SD")
        r_artists_map["Outer mean±SD"] = outer_reseated_mean_lines + outer_reseated_bands
        r_actives.append(True)
    if r_labels:
        r_check = CheckButtons(r_ax, labels=r_labels, actives=r_actives)
        try:
            for txt in r_check.labels:
                txt.set_fontsize(10)
        except Exception:
            pass
        try:
            r_ax.set_title("Reseated", fontsize=10, pad=2)
        except Exception:
            pass

        def on_toggle_r(label: str) -> None:
            try:
                idx = r_labels.index(label)
            except ValueError:
                return
            visible = r_check.get_status()[idx]
            for artist in r_artists_map.get(label, []):
                artist.set_visible(visible)
            for a in axes.ravel():
                a.figure.canvas.draw_idle()

        r_check.on_clicked(on_toggle_r)

    # Hover tooltips showing source filename for single-run lines
    run_lines = mounted_lines + reseated_lines + inner_mounted_lines + inner_reseated_lines + outer_mounted_lines + outer_reseated_lines + truth_mounted_lines + truth_reseated_lines
    if run_lines:
        cursor = mplcursors.cursor(run_lines, hover=True)

        @cursor.connect("add")
        def on_add(sel):
            line = sel.artist
            source = getattr(line, "_source_path", None)
            if not source:
                source = line.get_label() or ""
            sel.annotation.set_text(str(source))

        # Separate hover toggle above the line toggles (not in the same box)
        hax = fig.add_axes([0.86, 0.90, 0.12, 0.08])
        hover_check = CheckButtons(hax, labels=["Hover filenames"], actives=[True])
        try:
            for txt in hover_check.labels:
                txt.set_fontsize(10)
        except Exception:
            pass

        def on_hover_toggle(_label: str) -> None:
            enabled = hover_check.get_status()[0]
            try:
                cursor.enabled = enabled
                if not enabled:
                    # Hide any existing annotations when disabling (robust across mplcursors versions)
                    try:
                        for ann in list(getattr(cursor, "annotations", [])):
                            ann.set_visible(False)
                    except Exception:
                        pass
                    try:
                        for sel in list(getattr(cursor, "selections", [])):
                            try:
                                sel.annotation.set_visible(False)
                            except Exception:
                                pass
                    except Exception:
                        pass
                for a in axes.ravel():
                    a.figure.canvas.draw_idle()
            except Exception:
                pass

        hover_check.on_clicked(on_hover_toggle)

    fig.suptitle("Load Cell Sensitivity: Mounted (blue) vs Reseated (red) vs Truth (black/gray)")
    # Show interactively instead of saving for now
    plt.show()


def main() -> None:
    paths = Paths()
    params = Params()

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

    # Metrics are computed in-memory as needed for plots; nothing is written to disk.


if __name__ == "__main__":
    main()




