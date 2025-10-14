import os
from typing import List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import mplcursors
from matplotlib.widgets import CheckButtons

from overlay_mounted_vs_reseated import (
    Paths,
    Params,
    load_group,
    preprocess_signal,
    rotate_group_about_z,
    align_group_by_peak,
    align_group_by_local_peak,
    align_run,
    align_sets_by_first_x_peak,
    compute_group_stats_aligned,
    smooth_for_alignment,
)


def _pair_for_axis_bz_x(axis: str) -> Tuple[str, str]:
    # measured axis stays the same; truth axis forced to bz for x-axis
    if axis == "x":
        return "x", "bz"
    if axis == "y":
        return "y", "bz"
    if axis == "z":
        return "z", "bz"
    if axis == "mag":
        return "mag", "bz"
    return axis, "bz"


def plot_measured_vs_truth_bz_x(
    mounted: List[pd.DataFrame],
    reseated: List[pd.DataFrame],
    params: Params,
) -> None:
    sns.set_style("whitegrid")
    cols = ["x", "y", "z", "mag"]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes_map = {"x": (0, 0), "y": (0, 1), "z": (1, 0), "mag": (1, 1)}

    mounted_lines: list = []
    reseated_lines: list = []
    mounted_dots: list = []
    reseated_dots: list = []
    mounted_mean_lines: list = []
    reseated_mean_lines: list = []
    mounted_bands: list = []
    reseated_bands: list = []

    for col in cols:
        meas_col, truth_col = _pair_for_axis_bz_x(col)
        r, c = axes_map[col]
        ax = axes[r][c]
        ax.set_title(f"{meas_col.upper()} vs BZ")
        ax.set_xlabel("BZ")
        ax.set_ylabel(meas_col.upper())

        tx_min = None
        tx_max = None
        my_min = None
        my_max = None

        mounted_pairs_x: list = []
        mounted_pairs_y: list = []
        reseated_pairs_x: list = []
        reseated_pairs_y: list = []

        # Mounted: lines sorted by bz, optional dots (hidden by default)
        for df in mounted:
            if meas_col not in df.columns or truth_col not in df.columns:
                continue
            x_truth = df[truth_col].to_numpy()
            y_meas = df[meas_col].to_numpy()
            sort_idx = np.argsort(df["bz"].to_numpy()) if "bz" in df.columns else np.argsort(x_truth)
            n = min(len(x_truth), len(y_meas))
            if n <= 0:
                continue
            x_arr = x_truth[:n].astype(float)[sort_idx[:n]]
            y_arr = y_meas[:n].astype(float)[sort_idx[:n]]
            line, = ax.plot(x_arr, y_arr, color="#1f77b4", alpha=0.7, linewidth=1.0)
            source = df.get("source_path", None)
            try:
                line._source_path = str(source if isinstance(source, str) else (source.iloc[0] if source is not None else ""))
            except Exception:
                line._source_path = ""
            mounted_lines.append(line)
            # dots
            dx = x_truth[:n].astype(float)
            dy = y_meas[:n].astype(float)
            sc = ax.scatter(dx, dy, s=4, alpha=0.25, color="#1f77b4")
            try:
                sc._source_path = str(source if isinstance(source, str) else (source.iloc[0] if source is not None else ""))
            except Exception:
                sc._source_path = ""
            sc.set_visible(False)
            mounted_dots.append(sc)

            mounted_pairs_x.append(x_truth.astype(float)[:n])
            mounted_pairs_y.append(y_meas.astype(float)[:n])
            cur_tx_min = float(np.nanmin(x_arr))
            cur_tx_max = float(np.nanmax(x_arr))
            cur_my_min = float(np.nanmin(y_arr))
            cur_my_max = float(np.nanmax(y_arr))
            tx_min = cur_tx_min if tx_min is None else min(tx_min, cur_tx_min)
            tx_max = cur_tx_max if tx_max is None else max(tx_max, cur_tx_max)
            my_min = cur_my_min if my_min is None else min(my_min, cur_my_min)
            my_max = cur_my_max if my_max is None else max(my_max, cur_my_max)

        # Reseated
        for df in reseated:
            if meas_col not in df.columns or truth_col not in df.columns:
                continue
            x_truth = df[truth_col].to_numpy()
            y_meas = df[meas_col].to_numpy()
            sort_idx = np.argsort(df["bz"].to_numpy()) if "bz" in df.columns else np.argsort(x_truth)
            n = min(len(x_truth), len(y_meas))
            if n <= 0:
                continue
            x_arr = x_truth[:n].astype(float)[sort_idx[:n]]
            y_arr = y_meas[:n].astype(float)[sort_idx[:n]]
            line, = ax.plot(x_arr, y_arr, color="#d62728", alpha=0.7, linewidth=1.0)
            source = df.get("source_path", None)
            try:
                line._source_path = str(source if isinstance(source, str) else (source.iloc[0] if source is not None else ""))
            except Exception:
                line._source_path = ""
            reseated_lines.append(line)
            # dots
            dx = x_truth[:n].astype(float)
            dy = y_meas[:n].astype(float)
            sc = ax.scatter(dx, dy, s=4, alpha=0.25, color="#d62728")
            try:
                sc._source_path = str(source if isinstance(source, str) else (source.iloc[0] if source is not None else ""))
            except Exception:
                sc._source_path = ""
            sc.set_visible(False)
            reseated_dots.append(sc)

            reseated_pairs_x.append(x_truth.astype(float)[:n])
            reseated_pairs_y.append(y_meas.astype(float)[:n])
            cur_tx_min = float(np.nanmin(x_arr))
            cur_tx_max = float(np.nanmax(x_arr))
            cur_my_min = float(np.nanmin(y_arr))
            cur_my_max = float(np.nanmax(y_arr))
            tx_min = cur_tx_min if tx_min is None else min(tx_min, cur_tx_min)
            tx_max = cur_tx_max if tx_max is None else max(tx_max, cur_tx_max)
            my_min = cur_my_min if my_min is None else min(my_min, cur_my_min)
            my_max = cur_my_max if my_max is None else max(my_max, cur_my_max)

        # Limits with padding
        if tx_min is None or tx_max is None:
            tx_min, tx_max = -1.0, 1.0
        if my_min is None or my_max is None:
            my_min, my_max = -1.0, 1.0
        if tx_min == tx_max:
            tx_max = tx_min + 1.0
        if my_min == my_max:
            my_max = my_min + 1.0
        pad_x = 0.05 * (tx_max - tx_min)
        pad_y = 0.05 * (my_max - my_min)
        ax.set_xlim(tx_min - pad_x, tx_max + pad_x)
        ax.set_ylim(my_min - pad_y, my_max + pad_y)

        # Per-set mean ± SD bands over bz bins (since truth is bz on x)
        try:
            n_bins = 60
            edges = np.linspace(tx_min, tx_max, n_bins + 1)
            centers = 0.5 * (edges[:-1] + edges[1:])
            if mounted_pairs_x and mounted_pairs_y:
                mx = np.concatenate(mounted_pairs_x)
                my = np.concatenate(mounted_pairs_y)
                m_means = np.full_like(centers, np.nan, dtype=float)
                m_stds = np.full_like(centers, np.nan, dtype=float)
                for i in range(n_bins):
                    mask = (mx >= edges[i]) & (mx < edges[i + 1])
                    vals = my[mask]
                    if vals.size:
                        m_means[i] = float(np.nanmean(vals))
                        m_stds[i] = float(np.nanstd(vals))
                m_line, = ax.plot(centers, m_means, color="#1f77b4", linewidth=2.0)
                mounted_mean_lines.append(m_line)
                m_band = ax.fill_between(centers, m_means - m_stds, m_means + m_stds, color="#1f77b4", alpha=0.15)
                mounted_bands.append(m_band)
            if reseated_pairs_x and reseated_pairs_y:
                rx = np.concatenate(reseated_pairs_x)
                ry = np.concatenate(reseated_pairs_y)
                r_means = np.full_like(centers, np.nan, dtype=float)
                r_stds = np.full_like(centers, np.nan, dtype=float)
                for i in range(n_bins):
                    mask = (rx >= edges[i]) & (rx < edges[i + 1])
                    vals = ry[mask]
                    if vals.size:
                        r_means[i] = float(np.nanmean(vals))
                        r_stds[i] = float(np.nanstd(vals))
                r_line, = ax.plot(centers, r_means, color="#d62728", linewidth=2.0)
                reseated_mean_lines.append(r_line)
                r_band = ax.fill_between(centers, r_means - r_stds, r_means + r_stds, color="#d62728", alpha=0.15)
                reseated_bands.append(r_band)
        except Exception:
            pass

    # Toggles
    fig.subplots_adjust(right=0.85)
    rax = fig.add_axes([0.87, 0.45, 0.12, 0.2])
    labels = []
    artists_map = {}
    actives = []

    if mounted_lines:
        labels.append("Mounted lines")
        artists_map["Mounted lines"] = mounted_lines
        actives.append(True)
    if reseated_lines:
        labels.append("Reseated lines")
        artists_map["Reseated lines"] = reseated_lines
        actives.append(True)
    if mounted_dots:
        labels.append("Mounted dots")
        artists_map["Mounted dots"] = mounted_dots
        actives.append(False)
    if reseated_dots:
        labels.append("Reseated dots")
        artists_map["Reseated dots"] = reseated_dots
        actives.append(False)
    if mounted_mean_lines or mounted_bands:
        labels.append("Mounted mean±SD")
        artists_map["Mounted mean±SD"] = mounted_mean_lines + mounted_bands
        actives.append(True)
    if reseated_mean_lines or reseated_bands:
        labels.append("Reseated mean±SD")
        artists_map["Reseated mean±SD"] = reseated_mean_lines + reseated_bands
        actives.append(True)

    if labels:
        check = CheckButtons(rax, labels=labels, actives=actives)

        def on_toggle(label: str) -> None:
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

    # Hover
    run_artists = mounted_lines + reseated_lines + mounted_dots + reseated_dots
    if run_artists:
        cursor = mplcursors.cursor(run_artists, hover=True)

        @cursor.connect("add")
        def on_add(sel):
            artist = sel.artist
            source = getattr(artist, "_source_path", None) or ""
            sel.annotation.set_text(str(source))

        hax = fig.add_axes([0.87, 0.70, 0.12, 0.08])
        hover_check = CheckButtons(hax, labels=["Hover filenames"], actives=[True])

        def on_hover_toggle(_label: str) -> None:
            enabled = hover_check.get_status()[0]
            try:
                cursor.enabled = enabled
                if not enabled:
                    try:
                        for ann in list(cursor.annotations):
                            ann.set_visible(False)
                    except Exception:
                        pass
                for a in axes.ravel():
                    a.figure.canvas.draw_idle()
            except Exception:
                pass

        hover_check.on_clicked(on_hover_toggle)

    fig.suptitle("Measured vs Truth (X-axis: BZ): Mounted (blue) vs Reseated (red)")
    plt.show()


def main() -> None:
    paths = Paths()
    params = Params()

    mounted_raw = load_group(paths.mounted_glob, params.axis_mapping)
    reseated_raw = load_group(paths.reseated_glob, params.axis_mapping)

    mounted_prep = [preprocess_signal(df, params) for df in mounted_raw]
    reseated_prep = [preprocess_signal(df, params) for df in reseated_raw]

    # Alignment and rotation like the original script
    if params.alignment_method == "peak":
        mounted_aligned = align_group_by_peak(mounted_prep, params, signal="mag")
        reseated_aligned = align_group_by_peak(reseated_prep, params, signal="mag")
    elif params.alignment_method == "local_peak":
        mounted_aligned = align_group_by_local_peak(mounted_prep, params)
        reseated_aligned = align_group_by_local_peak(reseated_prep, params)
    else:
        mounted_aligned = [align_run(df, params) for df in mounted_prep]
        reseated_aligned = [align_run(df, params) for df in reseated_prep]

    mounted_rot = rotate_group_about_z(mounted_aligned, params.rotation_deg_z)
    reseated_rot = rotate_group_about_z(reseated_aligned, params.rotation_deg_z)

    mounted_adj, reseated_adj = align_sets_by_first_x_peak(mounted_rot, reseated_rot, params)

    # Trim by truth z slope (same as original)
    def trim_by_truth_z_slope(m_list: List[pd.DataFrame], r_list: List[pd.DataFrame], slope_threshold: float = -1.0, smooth_window: int = 10) -> tuple[List[pd.DataFrame], List[pd.DataFrame]]:
        try:
            combined: List[pd.DataFrame] = []
            for df in (m_list + r_list):
                if df is not None and ("frame_aligned" in df.columns) and ("bz" in df.columns):
                    combined.append(df[["frame_aligned", "bz"]].copy())
            if not combined:
                return m_list, r_list
            stats, grid = compute_group_stats_aligned(combined, ["bz"], x_col="frame_aligned")
            if not stats or grid.size == 0 or "bz" not in stats:
                return m_list, r_list
            z_mean = stats["bz"]["mean"].to_numpy()
            z_smooth = smooth_for_alignment(z_mean, smooth_window)
            slopes = np.diff(z_smooth)
            idx = None
            for i in range(slopes.size):
                if np.isfinite(slopes[i]) and slopes[i] <= slope_threshold:
                    idx = i + 1
                    break
            if idx is None:
                return m_list, r_list
            cutoff_frame = int(grid[idx])
            def cut_list(lst: List[pd.DataFrame]) -> List[pd.DataFrame]:
                out: List[pd.DataFrame] = []
                for d in lst:
                    if "frame_aligned" in d.columns:
                        out.append(d.loc[d["frame_aligned"] <= cutoff_frame].reset_index(drop=True).copy())
                    else:
                        out.append(d)
                return out
            return cut_list(m_list), cut_list(r_list)
        except Exception:
            return m_list, r_list

    mounted_trim, reseated_trim = trim_by_truth_z_slope(mounted_adj, reseated_adj)

    plot_measured_vs_truth_bz_x(mounted_trim, reseated_trim, params)


if __name__ == "__main__":
    main()


