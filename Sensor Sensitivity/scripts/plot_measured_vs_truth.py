import os
from typing import List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import mplcursors
from matplotlib.widgets import CheckButtons, Button

# Reuse pipeline helpers and configuration from the overlay module
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


def _pair_for_axis(axis: str) -> Tuple[str, str]:
    # measured, truth
    if axis == "x":
        return "x", "bx"
    if axis == "y":
        return "y", "by"
    if axis == "z":
        return "z", "bz"
    if axis == "mag":
        return "mag", "bmag"
    return axis, axis

def _pair_for_axis_labels(axis: str) -> Tuple[str, str]:
    # measured, truth
    if axis == "x":
        return "$F_x$ [N]", r"$B_x\ [\mu\mathrm{T}]$"
    if axis == "y":
        return "$F_y$ [N]", r"$B_y\ [\mu\mathrm{T}]$"
    if axis == "z":
        return "$F_z$ [N]", r"$B_z\ [\mu\mathrm{T}]$"
    if axis == "mag":
        return "$|F|$ [N]", r"$|B|\ [\mu\mathrm{T}]$"
    return axis, axis


def plot_measured_vs_truth(
    mounted: List[pd.DataFrame],
    reseated: List[pd.DataFrame],
    params: Params,
) -> None:
    sns.set_style("whitegrid")
    cols = ["x", "y", "z", "mag"]
    fig, axes = plt.subplots(2, 2, figsize=(16, 9))
    axes_map = {"x": (0, 0), "y": (0, 1), "z": (1, 0), "mag": (1, 1)}
    # Collect CheckButtons groups for global control
    check_groups: list = []

    # Sum variant artists
    mounted_lines_sum: list = []
    reseated_lines_sum: list = []
    mounted_dots_sum: list = []
    reseated_dots_sum: list = []
    mounted_mean_lines_sum: list = []
    reseated_mean_lines_sum: list = []
    mounted_bands_sum: list = []
    reseated_bands_sum: list = []
    # Inner variant artists
    mounted_lines_inner: list = []
    reseated_lines_inner: list = []
    mounted_dots_inner: list = []
    reseated_dots_inner: list = []
    mounted_mean_lines_inner: list = []
    reseated_mean_lines_inner: list = []
    mounted_bands_inner: list = []
    reseated_bands_inner: list = []
    # Outer variant artists
    mounted_lines_outer: list = []
    reseated_lines_outer: list = []
    mounted_dots_outer: list = []
    reseated_dots_outer: list = []
    mounted_mean_lines_outer: list = []
    reseated_mean_lines_outer: list = []
    mounted_bands_outer: list = []
    reseated_bands_outer: list = []

    for col in cols:
        meas_col, truth_col = _pair_for_axis(col)
        meas_label, truth_label = _pair_for_axis_labels(col)
        r, c = axes_map[col]
        ax = axes[r][c]
        # ax.set_title(f"{meas_col.upper()} vs {truth_col.upper()}")
        ax.set_xlabel(meas_label, fontsize=18)
        ax.set_ylabel(truth_label, fontsize=18)
        ax.tick_params(axis='both', which='major', labelsize=12)

        # Track per-axis ranges separately for better fit
        tx_min = None  # truth x-axis min
        tx_max = None  # truth x-axis max
        my_min = None  # measured y-axis min
        my_max = None  # measured y-axis max

        # Collect for mean bands (per variant)
        mounted_pairs_x_sum: list = []
        mounted_pairs_y_sum: list = []
        reseated_pairs_x_sum: list = []
        reseated_pairs_y_sum: list = []
        mounted_pairs_x_inner: list = []
        mounted_pairs_y_inner: list = []
        reseated_pairs_x_inner: list = []
        reseated_pairs_y_inner: list = []
        mounted_pairs_x_outer: list = []
        mounted_pairs_y_outer: list = []
        reseated_pairs_x_outer: list = []
        reseated_pairs_y_outer: list = []

        # Plot mounted (sum; as lines sorted by bmag) and optional dots (raw)
        for df in mounted:
            if meas_col not in df.columns or truth_col not in df.columns:
                continue
            x_truth = df[truth_col].to_numpy()
            y_meas = df[meas_col].to_numpy()
            sort_idx = np.argsort(x_truth)
            n = min(len(x_truth), len(y_meas))
            if n <= 0:
                continue
            x_arr = x_truth[:n].astype(float)[sort_idx[:n]]
            y_arr = y_meas[:n].astype(float)[sort_idx[:n]]
            line, = ax.plot(x_arr, y_arr, color="#1f77b4", alpha=0.7, linewidth=1.0, linestyle="-")
            source = df.get("source_path", None)
            try:
                line._source_path = str(source if isinstance(source, str) else (source.iloc[0] if source is not None else ""))
            except Exception:
                line._source_path = ""
            mounted_lines_sum.append(line)
            # raw dots (default hidden)
            dx = x_truth[:n].astype(float)
            dy = y_meas[:n].astype(float)
            sc = ax.scatter(dx, dy, s=4, alpha=0.25, color="#1f77b4")
            try:
                sc._source_path = str(source if isinstance(source, str) else (source.iloc[0] if source is not None else ""))
            except Exception:
                sc._source_path = ""
            sc.set_visible(False)
            mounted_dots_sum.append(sc)
            # accumulate for mean bands
            mounted_pairs_x_sum.append(x_truth.astype(float)[:n])
            mounted_pairs_y_sum.append(y_meas.astype(float)[:n])
            cur_tx_min = float(np.nanmin(x_arr))
            cur_tx_max = float(np.nanmax(x_arr))
            cur_my_min = float(np.nanmin(y_arr))
            cur_my_max = float(np.nanmax(y_arr))
            tx_min = cur_tx_min if tx_min is None else min(tx_min, cur_tx_min)
            tx_max = cur_tx_max if tx_max is None else max(tx_max, cur_tx_max)
            my_min = cur_my_min if my_min is None else min(my_min, cur_my_min)
            my_max = cur_my_max if my_max is None else max(my_max, cur_my_max)

        # Mounted INNER variant
        meas_col_inner = f"{meas_col}_inner" if meas_col != "mag" else "mag_inner"
        for df in mounted:
            if meas_col_inner not in df.columns or truth_col not in df.columns:
                continue
            x_truth = df[truth_col].to_numpy()
            y_meas = df[meas_col_inner].to_numpy()
            sort_idx = np.argsort(x_truth)
            n = min(len(x_truth), len(y_meas))
            if n <= 0:
                continue
            x_arr = x_truth[:n].astype(float)[sort_idx[:n]]
            y_arr = y_meas[:n].astype(float)[sort_idx[:n]]
            line, = ax.plot(x_arr, y_arr, color="#1f77b4", alpha=0.7, linewidth=1.0, linestyle="--")
            source = df.get("source_path", None)
            try:
                line._source_path = str(source if isinstance(source, str) else (source.iloc[0] if source is not None else ""))
            except Exception:
                line._source_path = ""
            mounted_lines_inner.append(line)
            dx = x_truth[:n].astype(float)
            dy = y_meas[:n].astype(float)
            sc = ax.scatter(dx, dy, s=4, alpha=0.25, color="#1f77b4")
            try:
                sc._source_path = str(source if isinstance(source, str) else (source.iloc[0] if source is not None else ""))
            except Exception:
                sc._source_path = ""
            sc.set_visible(False)
            mounted_dots_inner.append(sc)
            mounted_pairs_x_inner.append(x_truth.astype(float)[:n])
            mounted_pairs_y_inner.append(y_meas.astype(float)[:n])

        # Mounted OUTER variant
        meas_col_outer = f"{meas_col}_outer" if meas_col != "mag" else "mag_outer"
        for df in mounted:
            if meas_col_outer not in df.columns or truth_col not in df.columns:
                continue
            x_truth = df[truth_col].to_numpy()
            y_meas = df[meas_col_outer].to_numpy()
            sort_idx = np.argsort(x_truth)
            n = min(len(x_truth), len(y_meas))
            if n <= 0:
                continue
            x_arr = x_truth[:n].astype(float)[sort_idx[:n]]
            y_arr = y_meas[:n].astype(float)[sort_idx[:n]]
            line, = ax.plot(x_arr, y_arr, color="#1f77b4", alpha=0.7, linewidth=1.0, linestyle=":")
            source = df.get("source_path", None)
            try:
                line._source_path = str(source if isinstance(source, str) else (source.iloc[0] if source is not None else ""))
            except Exception:
                line._source_path = ""
            mounted_lines_outer.append(line)
            dx = x_truth[:n].astype(float)
            dy = y_meas[:n].astype(float)
            sc = ax.scatter(dx, dy, s=4, alpha=0.25, color="#1f77b4")
            try:
                sc._source_path = str(source if isinstance(source, str) else (source.iloc[0] if source is not None else ""))
            except Exception:
                sc._source_path = ""
            sc.set_visible(False)
            mounted_dots_outer.append(sc)
            mounted_pairs_x_outer.append(x_truth.astype(float)[:n])
            mounted_pairs_y_outer.append(y_meas.astype(float)[:n])

        # Plot reseated (sum)
        for df in reseated:
            if meas_col not in df.columns or truth_col not in df.columns:
                continue
            x_truth = df[truth_col].to_numpy()
            y_meas = df[meas_col].to_numpy()
            sort_idx = np.argsort(x_truth)
            n = min(len(x_truth), len(y_meas))
            if n <= 0:
                continue
            x_arr = x_truth[:n].astype(float)[sort_idx[:n]]
            y_arr = y_meas[:n].astype(float)[sort_idx[:n]]
            line, = ax.plot(x_arr, y_arr, color="#d62728", alpha=0.7, linewidth=1.0, linestyle="-")
            source = df.get("source_path", None)
            try:
                line._source_path = str(source if isinstance(source, str) else (source.iloc[0] if source is not None else ""))
            except Exception:
                line._source_path = ""
            reseated_lines_sum.append(line)
            # raw dots (default hidden)
            dx = x_truth[:n].astype(float)
            dy = y_meas[:n].astype(float)
            sc = ax.scatter(dx, dy, s=4, alpha=0.25, color="#d62728")
            try:
                sc._source_path = str(source if isinstance(source, str) else (source.iloc[0] if source is not None else ""))
            except Exception:
                sc._source_path = ""
            sc.set_visible(False)
            reseated_dots_sum.append(sc)
            reseated_pairs_x_sum.append(x_truth.astype(float)[:n])
            reseated_pairs_y_sum.append(y_meas.astype(float)[:n])
            cur_tx_min = float(np.nanmin(x_arr))
            cur_tx_max = float(np.nanmax(x_arr))
            cur_my_min = float(np.nanmin(y_arr))
            cur_my_max = float(np.nanmax(y_arr))
            tx_min = cur_tx_min if tx_min is None else min(tx_min, cur_tx_min)
            tx_max = cur_tx_max if tx_max is None else max(tx_max, cur_tx_max)
            my_min = cur_my_min if my_min is None else min(my_min, cur_my_min)
            my_max = cur_my_max if my_max is None else max(my_max, cur_my_max)

        # Reseated INNER
        for df in reseated:
            if meas_col_inner not in df.columns or truth_col not in df.columns:
                continue
            x_truth = df[truth_col].to_numpy()
            y_meas = df[meas_col_inner].to_numpy()
            sort_idx = np.argsort(x_truth)
            n = min(len(x_truth), len(y_meas))
            if n <= 0:
                continue
            x_arr = x_truth[:n].astype(float)[sort_idx[:n]]
            y_arr = y_meas[:n].astype(float)[sort_idx[:n]]
            line, = ax.plot(x_arr, y_arr, color="#d62728", alpha=0.7, linewidth=1.0, linestyle="--")
            source = df.get("source_path", None)
            try:
                line._source_path = str(source if isinstance(source, str) else (source.iloc[0] if source is not None else ""))
            except Exception:
                line._source_path = ""
            reseated_lines_inner.append(line)
            dx = x_truth[:n].astype(float)
            dy = y_meas[:n].astype(float)
            sc = ax.scatter(dx, dy, s=4, alpha=0.25, color="#d62728")
            try:
                sc._source_path = str(source if isinstance(source, str) else (source.iloc[0] if source is not None else ""))
            except Exception:
                sc._source_path = ""
            sc.set_visible(False)
            reseated_dots_inner.append(sc)
            reseated_pairs_x_inner.append(x_truth.astype(float)[:n])
            reseated_pairs_y_inner.append(y_meas.astype(float)[:n])

        # Reseated OUTER
        for df in reseated:
            if meas_col_outer not in df.columns or truth_col not in df.columns:
                continue
            x_truth = df[truth_col].to_numpy()
            y_meas = df[meas_col_outer].to_numpy()
            sort_idx = np.argsort(x_truth)
            n = min(len(x_truth), len(y_meas))
            if n <= 0:
                continue
            x_arr = x_truth[:n].astype(float)[sort_idx[:n]]
            y_arr = y_meas[:n].astype(float)[sort_idx[:n]]
            line, = ax.plot(x_arr, y_arr, color="#d62728", alpha=0.7, linewidth=1.0, linestyle=":")
            source = df.get("source_path", None)
            try:
                line._source_path = str(source if isinstance(source, str) else (source.iloc[0] if source is not None else ""))
            except Exception:
                line._source_path = ""
            reseated_lines_outer.append(line)
            dx = x_truth[:n].astype(float)
            dy = y_meas[:n].astype(float)
            sc = ax.scatter(dx, dy, s=4, alpha=0.25, color="#d62728")
            try:
                sc._source_path = str(source if isinstance(source, str) else (source.iloc[0] if source is not None else ""))
            except Exception:
                sc._source_path = ""
            sc.set_visible(False)
            reseated_dots_outer.append(sc)
            reseated_pairs_x_outer.append(x_truth.astype(float)[:n])
            reseated_pairs_y_outer.append(y_meas.astype(float)[:n])

        # Axis limits with padding, no unity line, no forced aspect
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
        x_lo = tx_min - pad_x
        x_hi = tx_max + pad_x
        y_lo = my_min - pad_y
        y_hi = my_max + pad_y
        ax.set_xlim(x_lo, x_hi)
        ax.set_ylim(y_lo, y_hi)

        # Per-set mean ± SD bands over truth bins for each variant
        try:
            n_bins = 60
            edges = np.linspace(tx_min, tx_max, n_bins + 1)
            centers = 0.5 * (edges[:-1] + edges[1:])
            # Mounted sum stats
            if mounted_pairs_x_sum and mounted_pairs_y_sum:
                mx = np.concatenate(mounted_pairs_x_sum)
                my = np.concatenate(mounted_pairs_y_sum)
                m_means = np.full_like(centers, np.nan, dtype=float)
                m_stds = np.full_like(centers, np.nan, dtype=float)
                for i in range(n_bins):
                    mask = (mx >= edges[i]) & (mx < edges[i + 1])
                    vals = my[mask]
                    if vals.size:
                        m_means[i] = float(np.nanmean(vals))
                        m_stds[i] = float(np.nanstd(vals))
                m_line, = ax.plot(centers, m_means, color="#1f77b4", linewidth=2.0, linestyle="-")
                mounted_mean_lines_sum.append(m_line)
                m_band = ax.fill_between(centers, m_means - m_stds, m_means + m_stds, color="#1f77b4", alpha=0.15)
                mounted_bands_sum.append(m_band)
            # Reseated sum stats
            if reseated_pairs_x_sum and reseated_pairs_y_sum:
                rx = np.concatenate(reseated_pairs_x_sum)
                ry = np.concatenate(reseated_pairs_y_sum)
                r_means = np.full_like(centers, np.nan, dtype=float)
                r_stds = np.full_like(centers, np.nan, dtype=float)
                for i in range(n_bins):
                    mask = (rx >= edges[i]) & (rx < edges[i + 1])
                    vals = ry[mask]
                    if vals.size:
                        r_means[i] = float(np.nanmean(vals))
                        r_stds[i] = float(np.nanstd(vals))
                r_line, = ax.plot(centers, r_means, color="#d62728", linewidth=2.0, linestyle="-")
                reseated_mean_lines_sum.append(r_line)
                r_band = ax.fill_between(centers, r_means - r_stds, r_means + r_stds, color="#d62728", alpha=0.15)
                reseated_bands_sum.append(r_band)

            # Mounted inner stats
            if mounted_pairs_x_inner and mounted_pairs_y_inner:
                mx = np.concatenate(mounted_pairs_x_inner)
                my = np.concatenate(mounted_pairs_y_inner)
                m_means = np.full_like(centers, np.nan, dtype=float)
                m_stds = np.full_like(centers, np.nan, dtype=float)
                for i in range(n_bins):
                    mask = (mx >= edges[i]) & (mx < edges[i + 1])
                    vals = my[mask]
                    if vals.size:
                        m_means[i] = float(np.nanmean(vals))
                        m_stds[i] = float(np.nanstd(vals))
                m_line, = ax.plot(centers, m_means, color="#1f77b4", linewidth=2.0, linestyle="--")
                mounted_mean_lines_inner.append(m_line)
                m_band = ax.fill_between(centers, m_means - m_stds, m_means + m_stds, color="#1f77b4", alpha=0.12)
                mounted_bands_inner.append(m_band)
            # Reseated inner stats
            if reseated_pairs_x_inner and reseated_pairs_y_inner:
                rx = np.concatenate(reseated_pairs_x_inner)
                ry = np.concatenate(reseated_pairs_y_inner)
                r_means = np.full_like(centers, np.nan, dtype=float)
                r_stds = np.full_like(centers, np.nan, dtype=float)
                for i in range(n_bins):
                    mask = (rx >= edges[i]) & (rx < edges[i + 1])
                    vals = ry[mask]
                    if vals.size:
                        r_means[i] = float(np.nanmean(vals))
                        r_stds[i] = float(np.nanstd(vals))
                r_line, = ax.plot(centers, r_means, color="#d62728", linewidth=2.0, linestyle="--")
                reseated_mean_lines_inner.append(r_line)
                r_band = ax.fill_between(centers, r_means - r_stds, r_means + r_stds, color="#d62728", alpha=0.12)
                reseated_bands_inner.append(r_band)

            # Mounted outer stats
            if mounted_pairs_x_outer and mounted_pairs_y_outer:
                mx = np.concatenate(mounted_pairs_x_outer)
                my = np.concatenate(mounted_pairs_y_outer)
                m_means = np.full_like(centers, np.nan, dtype=float)
                m_stds = np.full_like(centers, np.nan, dtype=float)
                for i in range(n_bins):
                    mask = (mx >= edges[i]) & (mx < edges[i + 1])
                    vals = my[mask]
                    if vals.size:
                        m_means[i] = float(np.nanmean(vals))
                        m_stds[i] = float(np.nanstd(vals))
                m_line, = ax.plot(centers, m_means, color="#1f77b4", linewidth=2.0, linestyle=":")
                mounted_mean_lines_outer.append(m_line)
                m_band = ax.fill_between(centers, m_means - m_stds, m_means + m_stds, color="#1f77b4", alpha=0.10)
                mounted_bands_outer.append(m_band)
            # Reseated outer stats
            if reseated_pairs_x_outer and reseated_pairs_y_outer:
                rx = np.concatenate(reseated_pairs_x_outer)
                ry = np.concatenate(reseated_pairs_y_outer)
                r_means = np.full_like(centers, np.nan, dtype=float)
                r_stds = np.full_like(centers, np.nan, dtype=float)
                for i in range(n_bins):
                    mask = (rx >= edges[i]) & (rx < edges[i + 1])
                    vals = ry[mask]
                    if vals.size:
                        r_means[i] = float(np.nanmean(vals))
                        r_stds[i] = float(np.nanstd(vals))
                r_line, = ax.plot(centers, r_means, color="#d62728", linewidth=2.0, linestyle=":")
                reseated_mean_lines_outer.append(r_line)
                r_band = ax.fill_between(centers, r_means - r_stds, r_means + r_stds, color="#d62728", alpha=0.10)
                reseated_bands_outer.append(r_band)
        except Exception:
            pass

    # Interactive toggles - panel layout like overlay script
    fig.subplots_adjust(right=0.82, wspace=0.25, hspace=0.28)

    # Mounted panel
    m_ax = fig.add_axes([0.86, 0.20, 0.12, 0.22])
    m_labels = []
    m_map = {}
    m_act = []
    if mounted_lines_sum:
        m_labels.append("Sum lines")
        m_map["Sum lines"] = mounted_lines_sum
        m_act.append(True)
    if mounted_lines_inner:
        m_labels.append("Inner lines")
        m_map["Inner lines"] = mounted_lines_inner
        m_act.append(True)
    if mounted_lines_outer:
        m_labels.append("Outer lines")
        m_map["Outer lines"] = mounted_lines_outer
        m_act.append(True)
    if mounted_dots_sum:
        m_labels.append("Sum dots")
        m_map["Sum dots"] = mounted_dots_sum
        m_act.append(True)
    if mounted_dots_inner:
        m_labels.append("Inner dots")
        m_map["Inner dots"] = mounted_dots_inner
        m_act.append(True)
    if mounted_dots_outer:
        m_labels.append("Outer dots")
        m_map["Outer dots"] = mounted_dots_outer
        m_act.append(True)
    if mounted_mean_lines_sum or mounted_bands_sum:
        m_labels.append("Sum mean±SD")
        m_map["Sum mean±SD"] = mounted_mean_lines_sum + mounted_bands_sum
        m_act.append(True)
    if mounted_mean_lines_inner or mounted_bands_inner:
        m_labels.append("Inner mean±SD")
        m_map["Inner mean±SD"] = mounted_mean_lines_inner + mounted_bands_inner
        m_act.append(True)
    if mounted_mean_lines_outer or mounted_bands_outer:
        m_labels.append("Outer mean±SD")
        m_map["Outer mean±SD"] = mounted_mean_lines_outer + mounted_bands_outer
        m_act.append(True)
    if m_labels:
        m_check = CheckButtons(m_ax, labels=m_labels, actives=m_act)
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
            for artist in m_map.get(label, []):
                artist.set_visible(visible)
            for a in axes.ravel():
                a.figure.canvas.draw_idle()

        m_check.on_clicked(on_toggle_m)
        try:
            check_groups.append(m_check)
        except Exception:
            pass

    # Reseated panel
    r_ax = fig.add_axes([0.86, 0.45, 0.12, 0.22])
    r_labels = []
    r_map = {}
    r_act = []
    if reseated_lines_sum:
        r_labels.append("Sum lines")
        r_map["Sum lines"] = reseated_lines_sum
        r_act.append(True)
    if reseated_lines_inner:
        r_labels.append("Inner lines")
        r_map["Inner lines"] = reseated_lines_inner
        r_act.append(True)
    if reseated_lines_outer:
        r_labels.append("Outer lines")
        r_map["Outer lines"] = reseated_lines_outer
        r_act.append(True)
    if reseated_dots_sum:
        r_labels.append("Sum dots")
        r_map["Sum dots"] = reseated_dots_sum
        r_act.append(True)
    if reseated_dots_inner:
        r_labels.append("Inner dots")
        r_map["Inner dots"] = reseated_dots_inner
        r_act.append(True)
    if reseated_dots_outer:
        r_labels.append("Outer dots")
        r_map["Outer dots"] = reseated_dots_outer
        r_act.append(True)
    if reseated_mean_lines_sum or reseated_bands_sum:
        r_labels.append("Sum mean±SD")
        r_map["Sum mean±SD"] = reseated_mean_lines_sum + reseated_bands_sum
        r_act.append(True)
    if reseated_mean_lines_inner or reseated_bands_inner:
        r_labels.append("Inner mean±SD")
        r_map["Inner mean±SD"] = reseated_mean_lines_inner + reseated_bands_inner
        r_act.append(True)
    if reseated_mean_lines_outer or reseated_bands_outer:
        r_labels.append("Outer mean±SD")
        r_map["Outer mean±SD"] = reseated_mean_lines_outer + reseated_bands_outer
        r_act.append(True)
    if r_labels:
        r_check = CheckButtons(r_ax, labels=r_labels, actives=r_act)
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
            for artist in r_map.get(label, []):
                artist.set_visible(visible)
            for a in axes.ravel():
                a.figure.canvas.draw_idle()

        r_check.on_clicked(on_toggle_r)
        try:
            check_groups.append(r_check)
        except Exception:
            pass

    # Hover tooltips for lines and dots
    run_artists = (
        mounted_lines_sum + mounted_lines_inner + mounted_lines_outer +
        reseated_lines_sum + reseated_lines_inner + reseated_lines_outer +
        mounted_dots_sum + mounted_dots_inner + mounted_dots_outer +
        reseated_dots_sum + reseated_dots_inner + reseated_dots_outer
    )
    if run_artists:
        cursor = mplcursors.cursor(run_artists, hover=True)

        @cursor.connect("add")
        def on_add(sel):
            artist = sel.artist
            source = getattr(artist, "_source_path", None) or ""
            sel.annotation.set_text(str(source))

        hax = fig.add_axes([0.86, 0.70, 0.12, 0.08])
        hover_check = CheckButtons(hax, labels=["Hover filenames"], actives=[True])

        def on_hover_toggle(_label: str) -> None:
            enabled = hover_check.get_status()[0]
            try:
                cursor.enabled = enabled
                if not enabled:
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

    # Global "All OFF" button to disable all plot toggles at once
    try:
        all_off_ax = fig.add_axes([0.86, 0.10, 0.12, 0.05])
        all_off_btn = Button(all_off_ax, "All OFF")

        def on_all_off(_event) -> None:
            try:
                for chk in list(check_groups):
                    try:
                        statuses = list(chk.get_status())
                    except Exception:
                        continue
                    for i, st in enumerate(statuses):
                        if st:
                            try:
                                chk.set_active(i)
                            except Exception:
                                pass
                for a in axes.ravel():
                    a.figure.canvas.draw_idle()
            except Exception:
                pass

        all_off_btn.on_clicked(on_all_off)
    except Exception:
        pass

    fig.suptitle("Axioforce Load Cell Raw Data vs Reference Load Cell", fontsize=20, y=0.95)
    plt.show()


def main() -> None:
    paths = Paths()
    params = Params()

    mounted_raw = load_group(paths.mounted_glob, params.axis_mapping)
    reseated_raw = load_group(paths.reseated_glob, params.axis_mapping)

    mounted_prep = [preprocess_signal(df, params) for df in mounted_raw]
    reseated_prep = [preprocess_signal(df, params) for df in reseated_raw]

    # Align runs (same technique as analysis script)
    if params.alignment_method == "peak":
        mounted_aligned = align_group_by_peak(mounted_prep, params, signal="mag")
        reseated_aligned = align_group_by_peak(reseated_prep, params, signal="mag")
    elif params.alignment_method == "local_peak":
        mounted_aligned = align_group_by_local_peak(mounted_prep, params)
        reseated_aligned = align_group_by_local_peak(reseated_prep, params)
    else:
        mounted_aligned = [align_run(df, params) for df in mounted_prep]
        reseated_aligned = [align_run(df, params) for df in reseated_prep]

    # Rotate measured axes after alignment
    mounted_rot = rotate_group_about_z(mounted_aligned, params.rotation_deg_z)
    reseated_rot = rotate_group_about_z(reseated_aligned, params.rotation_deg_z)

    # Inter-set alignment by first local 'x' peak threshold window
    mounted_adj, reseated_adj = align_sets_by_first_x_peak(mounted_rot, reseated_rot, params)

    # Trim by truth z slope: find first aligned frame where smoothed slope <= -1, then drop later frames
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
                    idx = i + 1  # cutoff at the frame after this slope
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

    # Plot rotated, aligned, and trimmed data
    plot_measured_vs_truth(mounted_trim, reseated_trim, params)


if __name__ == "__main__":
    main()


