import os
import glob
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt


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
    local_peak_threshold: float = 1000.0  # threshold for local peak window on 'x'
    rotation_deg_z: float = -45.0  # rotate measured x/y/z about Z after alignment
    subtract_baseline: bool = False  # if True, subtract per-run median baseline
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
        # Be robust to CSVs with padded header names (e.g., "sum-x    ").
        # This keeps downstream axis_mapping lookups stable across export variants.
        try:
            df.columns = [str(c).strip() for c in df.columns]
        except Exception:
            pass
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
    if y.size == 0:
        return y
    w = int(window) if window is not None else 0
    if w <= 1:
        return y
    if (w % 2) == 0:
        w += 1
    kernel = np.ones(w, dtype=float) / float(w)
    return np.convolve(y.astype(float), kernel, mode="same")


def preprocess_signal(df: pd.DataFrame, params: Params) -> pd.DataFrame:
    out = df.copy()
    # Optional baseline subtraction per run
    if params.subtract_baseline:
        if "time" in out.columns:
            t0 = out["time"].iloc[0]
            baseline_mask = out["time"] <= (t0 + params.baseline_seconds * 1000.0)
        else:
            n = max(1, int(0.5 * len(out)))
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
    return int(idx) if mag[idx] >= threshold else 0


def align_run(df: pd.DataFrame, params: Params) -> pd.DataFrame:
    if params.alignment_method == "frame":
        return df.reset_index(drop=True)
    if params.alignment_method == "event":
        onset = find_onset_index(df["mag"].to_numpy(), params.onset_threshold)
        if onset <= 0:
            return df.reset_index(drop=True)
        aligned = df.iloc[onset:].reset_index(drop=True).copy()
        if "time" in aligned.columns:
            t0 = aligned["time"].iloc[0]
            aligned["time_rel_ms"] = aligned["time"] - t0
        return aligned
    raise ValueError(f"Unknown alignment_method: {params.alignment_method}")


def align_group_by_peak(dfs: List[pd.DataFrame], params: Params, signal: Optional[str] = None) -> List[pd.DataFrame]:
    if not dfs:
        return dfs
    sig = signal if signal is not None else params.alignment_signal
    if sig not in dfs[0].columns:
        sig = "mag" if "mag" in dfs[0].columns else list(dfs[0].columns)[0]
    peak_indices: List[int] = []
    for df in dfs:
        y = df[sig].to_numpy()
        if y.size == 0:
            peak_indices.append(0)
        else:
            peak_indices.append(int(np.nanargmax(np.abs(y))))
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
    above = x > threshold
    if not np.any(above):
        return None
    start_idx = None
    for i in range(1, len(x)):
        if not above[i - 1] and above[i]:
            start_idx = i
            break
    if start_idx is None:
        return None
    end_idx = None
    for j in range(start_idx + 1, len(x)):
        if above[j - 1] and not above[j]:
            end_idx = j
            break
    if end_idx is None:
        return None
    if end_idx <= start_idx:
        return None
    window = x[start_idx:end_idx]
    peak_rel = int(np.nanargmax(window))
    return start_idx + peak_rel


def align_group_by_local_peak(dfs: List[pd.DataFrame], params: Params, threshold: Optional[float] = None) -> List[pd.DataFrame]:
    if not dfs:
        return dfs
    thr = params.local_peak_threshold if threshold is None else threshold
    peak_indices: List[int] = []
    for df in dfs:
        x = df["x"].to_numpy()
        x_for_peak = smooth_for_alignment(x, params.align_smoothing_window) if params.align_smoothing_enabled else x
        idx = find_local_peak_window_index(x_for_peak, thr)
        if idx is None:
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
        if all(col in out.columns for col in ("x_inner", "y_inner", "z_inner")):
            xyz_i = out[["x_inner", "y_inner", "z_inner"]].to_numpy(dtype=float)
            rot_i = xyz_i @ Rz.T
            out[["x_inner", "y_inner", "z_inner"]] = rot_i
            out["mag_inner"] = np.sqrt(np.sum(rot_i ** 2, axis=1))
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
    if not dfs:
        return {}, np.array([])
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
    for df in dfs:
        if "frame_aligned" not in df.columns:
            df["frame_aligned"] = np.arange(len(df), dtype=int)


def align_sets_by_first_x_peak(
    mounted: List[pd.DataFrame],
    reseated: List[pd.DataFrame],
    params: Params,
) -> Tuple[List[pd.DataFrame], List[pd.DataFrame]]:
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

    m_frame = int(m_grid[m_idx]) if m_grid.size and 0 <= m_idx < m_grid.size else 0
    r_frame = int(r_grid[r_idx]) if r_grid.size and 0 <= r_idx < r_grid.size else 0

    delta = r_frame - m_frame
    if delta == 0:
        return mounted, reseated

    out_reseated: List[pd.DataFrame] = []
    for df in reseated:
        out = df.copy()
        if "frame_aligned" in out.columns:
            out["frame_aligned"] = out["frame_aligned"].astype(int) - int(delta)
        out_reseated.append(out)
    return mounted, out_reseated


