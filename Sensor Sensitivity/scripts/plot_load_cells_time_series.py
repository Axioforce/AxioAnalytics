import os
import argparse
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib as mpl
mpl.use('qtagg')
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.widgets import CheckButtons, Button
from matplotlib.lines import Line2D
from pathlib import Path
import shutil
from functools import lru_cache

try:
    from matplotlib.backends.qt_compat import QtWidgets  # type: ignore
except Exception:
    QtWidgets = None

try:
    import mplcursors  # type: ignore
except Exception:
    mplcursors = None

from pipeline import (
    Params,
    load_group,
    preprocess_signal,
    rotate_group_about_z,
    align_group_by_peak,
    align_group_by_local_peak,
    align_run,
    compute_group_stats_aligned,
    smooth_for_alignment,
    find_local_peak_window_index,
)


def _pair_for_axis_labels(axis: str) -> Tuple[str, str]:
    if axis == "x":
        return "Time", "$F_x$ [N]"
    if axis == "y":
        return "Time", "$F_y$ [N]"
    if axis == "z":
        return "Time", "$F_z$ [N]"
    if axis == "mag":
        return "Time", "$|F|$ [N]"
    return "Time", axis


def _variant_column(base: str, variant: str) -> str:
    if variant == "sum":
        return base if base != "mag" else "mag"
    if variant == "inner":
        return f"{base}_inner" if base != "mag" else "mag_inner"
    if variant == "outer":
        return f"{base}_outer" if base != "mag" else "mag_outer"
    return base


def _truth_column(axis: str) -> str:
    if axis == "x":
        return "bx"
    if axis == "y":
        return "by"
    if axis == "z":
        return "bz"
    if axis == "mag":
        return "bmag"
    return axis


def _compute_aligned_mean_std(
    runs: List[pd.DataFrame],
    y_col: str,
    x_col: str = "frame_aligned",
) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    try:
        if not runs:
            return None
        stats, grid = compute_group_stats_aligned(runs, [y_col], x_col=x_col)
        if not stats or y_col not in stats or grid.size == 0:
            return None
        s = stats[y_col]
        return grid.astype(float), s["mean"].to_numpy(dtype=float), s["std"].to_numpy(dtype=float)
    except Exception:
        return None
def _compute_loaded_sumx_median(
    per_cell_runs: Dict[str, List[pd.DataFrame]],
    params: Params,
) -> Optional[float]:
    try:
        peak_frames: List[float] = []
        for runs in per_cell_runs.values():
            if not runs:
                continue
            res = _compute_aligned_mean_std(runs, _variant_column("x", "sum"), x_col="frame_aligned")
            if res is None:
                continue
            grid_x, means_x, _ = res
            if grid_x.size == 0 or means_x.size == 0:
                continue
            series = means_x.astype(float)
            try:
                idx = int(np.nanargmax(series)) if series.size else 0
            except Exception:
                idx = 0
            try:
                frame_val = float(grid_x[int(idx)])
            except Exception:
                frame_val = float(idx)
            peak_frames.append(frame_val)
        if peak_frames:
            return float(np.median(np.array(peak_frames, dtype=float)))
    except Exception:
        return None
    return None

def _align_group_by_x_smoothed_peak(dfs: List[pd.DataFrame], params: Params) -> List[pd.DataFrame]:
    if not dfs:
        return dfs
    peak_indices: List[int] = []
    for df in dfs:
        try:
            x = df["x"].to_numpy()
        except Exception:
            x = df.iloc[:, 0].to_numpy() if len(df.columns) else np.array([], dtype=float)
        x_for_peak = smooth_for_alignment(x, params.align_smoothing_window) if params.align_smoothing_enabled else x
        if x_for_peak.size == 0:
            peak_indices.append(0)
        else:
            peak_indices.append(int(np.nanargmax(x_for_peak)))
    ref_peak = int(np.median(peak_indices))
    aligned: List[pd.DataFrame] = []
    for df, p in zip(dfs, peak_indices):
        n = len(df)
        aligned_x = np.arange(n) - int(p) + int(ref_peak)
        out = df.copy()
        out["frame_aligned"] = aligned_x
        aligned.append(out)
    return aligned



def _derive_groups(cell_names: List[str]) -> Dict[str, List[str]]:
    groups: Dict[str, List[str]] = {"CH1-4": [], "CH5-8": [], "J": []}
    for name in cell_names:
        nm = str(name)
        nm_u = nm.upper()
        if nm_u.startswith("CH1") or nm_u.startswith("CH2") or nm_u.startswith("CH3") or nm_u.startswith("CH4"):
            groups["CH1-4"].append(name)
        if nm_u.startswith("CH5") or nm_u.startswith("CH6") or nm_u.startswith("CH7") or nm_u.startswith("CH8"):
            groups["CH5-8"].append(name)
        if nm_u.startswith("J"):
            groups["J"].append(name)
    return groups


@dataclass
class LCPaths:
    project_root: Path = Path(__file__).resolve().parent.parent
    default_root: Path = project_root / "Load_Cell_Runs"
    outputs_root: Path = project_root / "outputs" / "plots" / "load_cells_time_series"
    auto_source_root: Path = project_root / "Load_Cell_Spiral_test"


def _scan_load_cells(root: Path, only_cell: Optional[str] = None) -> Dict[str, str]:
    cells: Dict[str, str] = {}
    if only_cell:
        target = root / only_cell / "*.csv"
        cells[only_cell] = str(target)
        return cells
    for p in sorted((root).glob("*/")):
        name = p.name
        glob_path = p / "*.csv"
        cells[name] = str(glob_path)
    return cells


def _load_and_process_cell(cell_dir: Path, params: Params, use_abs_sum: bool = False) -> List[pd.DataFrame]:
    try:
        dfs = load_group(str(cell_dir / "*.csv"), params.axis_mapping)
    except Exception:
        dfs = []
    if not dfs:
        return []
    try:
        dfs_prep = [preprocess_signal(df, params) for df in dfs]
        method = getattr(params, "alignment_method", "x_peak_smoothed")
        if method == "x_peak_smoothed":
            dfs_aligned = _align_group_by_x_smoothed_peak(dfs_prep, params)
        elif method == "peak":
            dfs_aligned = align_group_by_peak(dfs_prep, params, signal="mag")
        elif method == "local_peak":
            dfs_aligned = align_group_by_local_peak(dfs_prep, params)
        else:
            dfs_aligned = [align_run(df, params) for df in dfs_prep]
        dfs_rot = rotate_group_about_z(dfs_aligned, params.rotation_deg_z)
        if use_abs_sum:
            try:
                for d in dfs_rot:
                    for base in ("x", "y", "z", "mag"):
                        inner_col = f"{base}_inner"
                        outer_col = f"{base}_outer"
                        if inner_col in d.columns and outer_col in d.columns:
                            try:
                                d[base] = d[inner_col].abs().astype(float) + d[outer_col].abs().astype(float)
                            except Exception:
                                pass
            except Exception:
                pass
        return dfs_rot
    except Exception:
        return []


def _compute_overall_cellwise_from_disk(params: Params, use_abs_sum: bool = False) -> Dict[str, Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]]]:
    root = LCPaths().default_root
    cells = _scan_load_cells(root)
    if not cells:
        return {}
    result: Dict[str, Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]]] = {}
    axes_list = ["x", "y", "z", "mag"]

    # Preload processed runs per cell
    cell_to_stats: Dict[str, Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]]] = {}
    for name in cells.keys():
        runs = _load_and_process_cell(root / name, params, use_abs_sum=use_abs_sum)
        if not runs:
            continue
        per_axis: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
        for axis in axes_list:
            y_col = _variant_column(axis, "sum")
            res = _compute_aligned_mean_std(runs, y_col, x_col="frame_aligned")
            if res is not None:
                per_axis[axis] = res
        if per_axis:
            cell_to_stats[name] = per_axis

    # Compute per-cell inter-cell alignment offsets using SUM-X mean curve
    cell_to_shift_frames: Dict[str, float] = {}
    try:
        peak_frames: List[float] = []
        per_cell_peak: Dict[str, float] = {}
        for name, per_axis in cell_to_stats.items():
            stat = per_axis.get("x")
            if not stat:
                continue
            grid_x, means_x, _ = stat
            if grid_x.size == 0 or means_x.size == 0:
                continue
            # Use raw SUM-X mean (no smoothing) for inter-cell alignment
            series = means_x.astype(float)
            try:
                idx = int(np.nanargmax(series)) if series.size else 0
            except Exception:
                idx = 0
            # Map index to actual frame value
            try:
                frame_val = float(grid_x[int(idx)])
            except Exception:
                frame_val = float(idx)
            per_cell_peak[name] = frame_val
            peak_frames.append(frame_val)
        if peak_frames:
            ref_frame = float(np.median(np.array(peak_frames, dtype=float)))
            for name, frame_val in per_cell_peak.items():
                # Positive shift means this cell occurs later; shift left by delta to align
                cell_to_shift_frames[name] = float(frame_val - ref_frame)
    except Exception:
        cell_to_shift_frames = {}

    # For each axis and variant, average across cells on intersection grid (after inter-cell shift)
    for axis in axes_list:
        axis_results: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
        for variant in ("sum", "inner", "outer"):
            v_col = _variant_column(axis, variant)
            # Gather per-cell stats for this axis
            grids: List[np.ndarray] = []
            means_list: List[np.ndarray] = []
            for name in cell_to_stats.keys():
                runs = _load_and_process_cell(root / name, params, use_abs_sum=use_abs_sum)
                if not runs or v_col not in runs[0].columns:
                    continue
                res = _compute_aligned_mean_std(runs, v_col, x_col="frame_aligned")
                if res is None:
                    continue
                grid, means, _stds = res
                # Apply inter-cell shift based on SUM-X peak alignment (if available)
                try:
                    shift = float(cell_to_shift_frames.get(name, 0.0))
                except Exception:
                    shift = 0.0
                grid_shifted = (grid.astype(float) - shift)
                grids.append(grid_shifted)
                means_list.append(means)
            if not grids:
                continue
            # Intersection grid after shifts
            try:
                gmin = max(float(g.min()) for g in grids)
                gmax = min(float(g.max()) for g in grids)
            except Exception:
                continue
            if gmax <= gmin:
                continue
            grid_common = np.arange(int(np.ceil(gmin)), int(np.floor(gmax)) + 1, dtype=float)
            # Reindex means to common grid
            stacked: List[np.ndarray] = []
            for grid, means in zip(grids, means_list):
                # Build mapping
                idx = {int(x): i for i, x in enumerate(grid.astype(int))}
                arr = np.full_like(grid_common, np.nan, dtype=float)
                for j, x in enumerate(grid_common.astype(int)):
                    i = idx.get(int(x))
                    if i is not None and 0 <= i < means.size:
                        arr[j] = float(means[i])
                stacked.append(arr)
            mat = np.vstack(stacked)
            mean_across = np.nanmean(mat, axis=0)
            std_across = np.nanstd(mat, axis=0)
            axis_results[variant] = (grid_common, mean_across, std_across)
        if axis_results:
            result[axis] = axis_results
    return result


def _compute_cell_shifts(
    per_cell_runs: Dict[str, List[pd.DataFrame]],
    params: Params,
    overall_ref_frame: Optional[float] = None,
) -> Dict[str, float]:
    # Compute per-cell inter-cell alignment offsets using SUM-X mean curve (no smoothing)
    cell_to_shift_frames: Dict[str, float] = {}
    try:
        peak_frames: List[float] = []
        per_cell_peak: Dict[str, float] = {}
        for name, runs in per_cell_runs.items():
            if not runs:
                continue
            y_col = _variant_column("x", "sum")
            res = _compute_aligned_mean_std(runs, y_col, x_col="frame_aligned")
            if res is None:
                continue
            grid_x, means_x, _ = res
            if grid_x.size == 0 or means_x.size == 0:
                continue
            series = means_x.astype(float)
            try:
                idx = int(np.nanargmax(series)) if series.size else 0
            except Exception:
                idx = 0
            try:
                frame_val = float(grid_x[int(idx)])
            except Exception:
                frame_val = float(idx)
            per_cell_peak[name] = frame_val
            peak_frames.append(frame_val)
        if peak_frames:
            if overall_ref_frame is not None and np.isfinite(overall_ref_frame):
                ref_frame = float(overall_ref_frame)
            else:
                ref_frame = float(np.median(np.array(peak_frames, dtype=float)))
            for name, frame_val in per_cell_peak.items():
                cell_to_shift_frames[name] = float(frame_val - ref_frame)
    except Exception:
        cell_to_shift_frames = {}
    return cell_to_shift_frames


def _select_load_cells_dialog(root: Path) -> List[str]:
    if QtWidgets is None:
        return []
    try:
        dialog = QtWidgets.QDialog()
        dialog.setWindowTitle("Add Load Cell to Plot")
        layout = QtWidgets.QVBoxLayout(dialog)
        label = QtWidgets.QLabel(f"Select one or more load cells from: {root}")
        layout.addWidget(label)
        list_widget = QtWidgets.QListWidget()
        list_widget.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)
        for p in sorted(root.glob("*/")):
            list_widget.addItem(p.name)
        layout.addWidget(list_widget)
        btns = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        layout.addWidget(btns)
        btns.accepted.connect(dialog.accept)
        btns.rejected.connect(dialog.reject)
        if dialog.exec() == QtWidgets.QDialog.Accepted:
            return [i.text() for i in list_widget.selectedItems()]
    except Exception:
        return []
    return []


def _attach_add_runs_to_project_button(
    fig,
    axes,
    params: Params,
    recompute_overall_and_refresh,
) -> None:
    try:
        manager = getattr(fig.canvas, "manager", None)
        toolbar = getattr(manager, "toolbar", None)
        if toolbar is None or QtWidgets is None:
            return

        try:
            toolbar.addSeparator()
        except Exception:
            pass
        action = toolbar.addAction("Add Runs to Project")

        def _on_add_runs() -> None:
            try:
                start_dir = str(LCPaths().default_root)
                files, _ = QtWidgets.QFileDialog.getOpenFileNames(
                    None,
                    "Select CSV files",
                    start_dir,
                    "CSV Files (*.csv)"
                )
                if not files:
                    return
                name, ok = QtWidgets.QInputDialog.getText(
                    None,
                    "Add Load Cell",
                    "Enter new load cell name:"
                )
                if not ok:
                    return
                name = str(name).strip()
                if not name:
                    return
                dest_dir = LCPaths().default_root / name
                dest_dir.mkdir(parents=True, exist_ok=True)
                for src in files:
                    try:
                        dst = dest_dir / os.path.basename(src)
                        shutil.copy2(src, dst)
                    except Exception:
                        pass
                recompute_overall_and_refresh()
            except Exception:
                pass

        action.triggered.connect(_on_add_runs)
    except Exception:
        pass


def _attach_add_cells_to_plot_button(
    fig,
    axes,
    per_cell_runs: Dict[str, List[pd.DataFrame]],
    params: Params,
    grouped_artists: Dict[Tuple[str, str], List],
    grouped_active: Dict[Tuple[str, str], bool],
    group_to_labels: Dict[str, List[str]],
    cell_to_color: Dict[str, Tuple[float, float, float]],
    palette,
    rebuild_toggles,
    use_abs_sum: bool,
    show_runs: bool,
    truth_toggles: bool,
    axes_truth_map: Dict[str, any],
    bands_enabled: List[bool],
) -> None:
    try:
        manager = getattr(fig.canvas, "manager", None)
        toolbar = getattr(manager, "toolbar", None)
        if toolbar is None or QtWidgets is None:
            return

        try:
            toolbar.addSeparator()
        except Exception:
            pass
        action = toolbar.addAction("Add Load Cell to Plot")

        def _on_add_cells() -> None:
            try:
                root = LCPaths().default_root
                selected = _select_load_cells_dialog(root)
                if not selected:
                    return

                # Assign colors for any new cells
                for name in selected:
                    if name not in cell_to_color:
                        idx = len(cell_to_color)
                        color = palette[idx % len(palette)]
                        cell_to_color[name] = color

                # Load and process selected cells not yet present
                for name in selected:
                    if name in per_cell_runs:
                        continue
                    runs = _load_and_process_cell(root / name, params, use_abs_sum=use_abs_sum)
                    if runs:
                        per_cell_runs[name] = runs

                # Recompute inter-cell shifts across all currently loaded cells
                cell_shifts_local = _compute_cell_shifts(per_cell_runs, params)

                # Remove existing per-cell artists from main axes and redraw with shifts
                try:
                    main_axes = [ax for ax in getattr(axes, "ravel", lambda: [])()] if hasattr(axes, "ravel") else list(axes)
                except Exception:
                    main_axes = []

                # Clear artists for all non-overall groups from main axes
                for gn, lbls in list(group_to_labels.items()):
                    if gn == "overall":
                        continue
                    for variant in list(lbls):
                        arts = grouped_artists.get((gn, variant), [])
                        new_keep = []
                        for a in arts:
                            try:
                                if getattr(a, "axes", None) in main_axes:
                                    a.remove()
                                else:
                                    new_keep.append(a)
                            except Exception:
                                new_keep.append(a)
                        grouped_artists[(gn, variant)] = new_keep

                # Redraw per-cell artists with applied shifts
                cols_local = ["x", "y", "z", "mag"]
                axes_map_local = {"x": (0, 0), "y": (0, 1), "z": (1, 0), "mag": (1, 1)}
                for col in cols_local:
                    r, c = axes_map_local[col]
                    ax = axes[r][c]
                    for cell_name, runs in per_cell_runs.items():
                        if not runs:
                            continue
                        color = cell_to_color.get(cell_name, (0.2, 0.4, 0.8))
                        shift = float(cell_shifts_local.get(cell_name, 0.0))
                        for variant, linestyle, alpha in (
                            ("sum", "-", 0.18),
                            ("inner", "--", 0.14),
                            ("outer", ":", 0.12),
                        ):
                            y_col = _variant_column(col, variant)
                            if y_col not in runs[0].columns:
                                continue
                            res = _compute_aligned_mean_std(runs, y_col, x_col="frame_aligned")
                            if res is None:
                                continue
                            grid, means, stds = res
                            try:
                                grid = (grid.astype(float) - shift)
                            except Exception:
                                pass
                            line, = ax.plot(grid, means, color=color, linewidth=2.0, linestyle=linestyle, label=f"{cell_name} {variant}")
                            band = ax.fill_between(grid, means - stds, means + stds, color=color, alpha=alpha)
                            key = (cell_name, variant)
                            grouped_artists.setdefault(key, []).extend([line, band])
                            default_visible = (variant == "sum")
                            grouped_active.setdefault(key, default_visible)
                            try:
                                line.set_visible(grouped_active.get(key, default_visible))
                                band.set_visible(grouped_active.get(key, default_visible))
                            except Exception:
                                pass
                            group_to_labels.setdefault(cell_name, [])
                            if variant not in group_to_labels[cell_name]:
                                group_to_labels[cell_name].append(variant)

                            if show_runs:
                                run_label = f"{variant} runs"
                                for df in runs:
                                    try:
                                        if y_col not in df.columns or "frame_aligned" not in df.columns:
                                            continue
                                        x = df["frame_aligned"].to_numpy(dtype=float)
                                        try:
                                            x = x - shift
                                        except Exception:
                                            pass
                                        y = df[y_col].to_numpy(dtype=float)
                                        line_run, = ax.plot(x, y, color=color, linewidth=0.9, linestyle="-", alpha=0.35)
                                        try:
                                            source = df.get("source_path", None)
                                            fname = str(source if isinstance(source, str) else (source.iloc[0] if source is not None else ""))
                                            line_run._source_path = fname
                                        except Exception:
                                            try:
                                                line_run._source_path = ""
                                            except Exception:
                                                pass
                                        grouped_artists.setdefault((cell_name, run_label), []).append(line_run)
                                        # Keep runs visibility off by default
                                        if (cell_name, run_label) not in grouped_active:
                                            grouped_active[(cell_name, run_label)] = False
                                        try:
                                            line_run.set_visible(grouped_active.get((cell_name, run_label), False))
                                        except Exception:
                                            pass
                                    except Exception:
                                        pass
                                if run_label not in group_to_labels[cell_name]:
                                    group_to_labels[cell_name].append(run_label)

                        # Truth mean and runs (optional)
                        if truth_toggles:
                            t_col = _truth_column(col)
                            if t_col in runs[0].columns:
                                res_t = _compute_aligned_mean_std(runs, t_col, x_col="frame_aligned")
                                if res_t is not None:
                                    grid_t, mean_t, std_t = res_t
                                    try:
                                        grid_t = (grid_t.astype(float) - shift)
                                    except Exception:
                                        pass
                                    # Plot truth on twin axis if available
                                    ax_truth = axes_truth_map.get(col)
                                    target_ax = ax_truth if truth_toggles and ax_truth is not None else ax
                                    line_t, = target_ax.plot(grid_t, mean_t, color=color, linewidth=2.0, linestyle="-.", label=f"{cell_name} truth")
                                    band_t = target_ax.fill_between(grid_t, mean_t - std_t, mean_t + std_t, color=color, alpha=0.12)
                                    key_t = (cell_name, "truth")
                                    grouped_artists.setdefault(key_t, []).extend([line_t, band_t])
                                    grouped_active.setdefault(key_t, True)
                                    try:
                                        line_t.set_visible(grouped_active.get(key_t, True))
                                    except Exception:
                                        pass
                                    group_to_labels.setdefault(cell_name, [])
                                    if "truth" not in group_to_labels[cell_name]:
                                        group_to_labels[cell_name].append("truth")

                                    if show_runs:
                                        run_label_t = "truth runs"
                                        for df in runs:
                                            try:
                                                if t_col not in df.columns or "frame_aligned" not in df.columns:
                                                    continue
                                                xt = df["frame_aligned"].to_numpy(dtype=float)
                                                try:
                                                    xt = xt - shift
                                                except Exception:
                                                    pass
                                                yt = df[t_col].to_numpy(dtype=float)
                                                ax_truth = axes_truth_map.get(col)
                                                target_ax = ax_truth if truth_toggles and ax_truth is not None else ax
                                                line_tr, = target_ax.plot(xt, yt, color=color, linewidth=0.9, linestyle="-.", alpha=0.35)
                                                try:
                                                    source = df.get("source_path", None)
                                                    fname = str(source if isinstance(source, str) else (source.iloc[0] if source is not None else ""))
                                                    line_tr._source_path = fname
                                                except Exception:
                                                    try:
                                                        line_tr._source_path = ""
                                                    except Exception:
                                                        pass
                                                grouped_artists.setdefault((cell_name, run_label_t), []).append(line_tr)
                                                if (cell_name, run_label_t) not in grouped_active:
                                                    grouped_active[(cell_name, run_label_t)] = False
                                                try:
                                                    line_tr.set_visible(grouped_active.get((cell_name, run_label_t), False))
                                                except Exception:
                                                    pass
                                            except Exception:
                                                pass
                                        if run_label_t not in group_to_labels[cell_name]:
                                            group_to_labels[cell_name].append(run_label_t)

                rebuild_toggles()
                fig.canvas.draw_idle()
            except Exception:
                pass

        action.triggered.connect(_on_add_cells)
    except Exception:
        pass


def plot_load_cells_time_series(
    per_cell_runs: Dict[str, List[pd.DataFrame]],
    params: Params,
    title_suffix: str = "",
    save_path: Optional[Path] = None,
    show: bool = True,
    use_abs_sum: bool = False,
    show_runs: bool = False,
    truth_toggles: bool = False,
    add_groups: bool = False,
) -> None:
    sns.set_style("whitegrid")
    cols = ["x", "y", "z", "mag"]
    # Create figure and set requested layout + fullscreen
    fig, axes = plt.subplots(2, 2, figsize=(16, 9))
    try:
        # Fullscreen (Qt backends) or best effort fallback
        mgr = getattr(fig.canvas, "manager", None)
        if mgr is not None:
            try:
                if hasattr(mgr, "window"):
                    mgr.window.showMaximized()
                else:
                    mgr.full_screen_toggle()
            except Exception:
                pass
        # User-specified default subplot geometry
        fig.subplots_adjust(top=0.89, bottom=0.085, left=0.065, right=0.805, hspace=0.255, wspace=0.355)
    except Exception:
        pass
    axes_map = {"x": (0, 0), "y": (0, 1), "z": (1, 0), "mag": (1, 1)}

    # Distinguish load cells by color
    cell_names = list(per_cell_runs.keys())
    palette = sns.color_palette("tab10", n_colors=10)
    cell_to_color: Dict[str, Tuple[float, float, float]] = {
        name: palette[i % len(palette)] for i, name in enumerate(cell_names)
    }

    grouped_artists: Dict[Tuple[str, str], List] = {}
    grouped_active: Dict[Tuple[str, str], bool] = {}
    group_to_labels: Dict[str, List[str]] = {}
    # Global toggle for bands; must be defined before any bands are created
    bands_enabled: List[bool] = [True]

    # Precompute OVERALL lines across disk cells
    overall_data = _compute_overall_cellwise_from_disk(params, use_abs_sum=use_abs_sum)

    # Optional: group-level lines from disk (using same logic as overall but grouping cell folders)
    group_overall: Dict[str, Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]]] = {}
    if add_groups:
        try:
            root = LCPaths().default_root
            cells = _scan_load_cells(root)
            groups = _derive_groups(list(cells.keys()))
            axes_list = ["x", "y", "z", "mag"]
            # Determine overall SUM-X reference frame once
            overall_ref = None
            try:
                overall_x = overall_data.get("x", {}).get("sum")
                if overall_x is not None:
                    og, omeans, _ostd = overall_x
                    if og.size and omeans.size:
                        overall_ref = float(og[int(np.nanargmax(omeans.astype(float)))])
            except Exception:
                overall_ref = None

            for gname, members in groups.items():
                if not members:
                    continue
                # Compute per-member shift using SUM-X mean peak (no smoothing), anchored to overall_ref if available, else group-median
                member_peak_frames: Dict[str, float] = {}
                for name in members:
                    runs_sumx = _load_and_process_cell(root / name, params, use_abs_sum=use_abs_sum)
                    if not runs_sumx:
                        continue
                    res_sumx = _compute_aligned_mean_std(runs_sumx, _variant_column("x", "sum"), x_col="frame_aligned")
                    if res_sumx is None:
                        continue
                    gx, mx, _sx = res_sumx
                    if gx.size and mx.size:
                        try:
                            idxp = int(np.nanargmax(mx.astype(float)))
                        except Exception:
                            idxp = 0
                        try:
                            member_peak_frames[name] = float(gx[idxp])
                        except Exception:
                            member_peak_frames[name] = float(idxp)
                # Choose reference
                if member_peak_frames:
                    if overall_ref is None or not np.isfinite(overall_ref):
                        try:
                            overall_ref_local = float(np.median(np.array(list(member_peak_frames.values()), dtype=float)))
                        except Exception:
                            overall_ref_local = None
                    else:
                        overall_ref_local = overall_ref
                else:
                    overall_ref_local = None

                g_axes: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
                for axis in axes_list:
                    v_map = {"sum": _variant_column(axis, "sum"), "inner": _variant_column(axis, "inner"), "outer": _variant_column(axis, "outer")}
                    grids: Dict[str, List[np.ndarray]] = {"sum": [], "inner": [], "outer": []}
                    means_lists: Dict[str, List[np.ndarray]] = {"sum": [], "inner": [], "outer": []}
                    for name in members:
                        runs = _load_and_process_cell(root / name, params, use_abs_sum=use_abs_sum)
                        if not runs:
                            continue
                        # Compute member shift (frame_val - ref)
                        try:
                            shift_val = 0.0
                            if name in member_peak_frames and overall_ref_local is not None and np.isfinite(overall_ref_local):
                                shift_val = float(member_peak_frames[name] - overall_ref_local)
                        except Exception:
                            shift_val = 0.0
                        for var_key, col in v_map.items():
                            if col not in runs[0].columns:
                                continue
                            res = _compute_aligned_mean_std(runs, col, x_col="frame_aligned")
                            if res is None:
                                continue
                            grid, means, _stds = res
                            try:
                                grid = (grid.astype(float) - float(shift_val))
                            except Exception:
                                grid = grid.astype(float)
                            grids[var_key].append(grid)
                            means_lists[var_key].append(means.astype(float))
                    for var_key in ("sum", "inner", "outer"):
                        if not grids[var_key]:
                            continue
                        try:
                            gmin = max(float(g.min()) for g in grids[var_key])
                            gmax = min(float(g.max()) for g in grids[var_key])
                        except Exception:
                            continue
                        if gmax <= gmin:
                            continue
                        grid_common = np.arange(int(np.ceil(gmin)), int(np.floor(gmax)) + 1, dtype=float)
                        stacked: List[np.ndarray] = []
                        for grid, means in zip(grids[var_key], means_lists[var_key]):
                            idx = {int(x): i for i, x in enumerate(grid.astype(int))}
                            arr = np.full_like(grid_common, np.nan, dtype=float)
                            for j, x in enumerate(grid_common.astype(int)):
                                i = idx.get(int(x))
                                if i is not None and 0 <= i < means.size:
                                    arr[j] = float(means[i])
                            stacked.append(arr)
                        mat = np.vstack(stacked)
                        mean_across = np.nanmean(mat, axis=0)
                        std_across = np.nanstd(mat, axis=0)
                        g_axes[f"{var_key}"] = (grid_common, mean_across, std_across)
                if g_axes:
                    group_overall[gname] = g_axes
        except Exception:
            group_overall = {}

    # Compute inter-cell shifts based on current per_cell_runs, aligning to overall SUM-X peak when available
    try:
        overall_x = overall_data.get("x", {}).get("sum")
        overall_ref = None
        if overall_x is not None:
            og, omeans, _ostd = overall_x
            if og.size and omeans.size:
                try:
                    overall_ref = float(og[int(np.nanargmax(omeans.astype(float)))])
                except Exception:
                    overall_ref = None
        cell_shifts = _compute_cell_shifts(per_cell_runs, params, overall_ref_frame=overall_ref)
    except Exception:
        cell_shifts = _compute_cell_shifts(per_cell_runs, params)

    # Prepare optional right-side truth axes per subplot
    axes_truth_map: Dict[str, any] = {}
    for col in cols:
        r, c = axes_map[col]
        ax = axes[r][c]
        x_label, y_label = _pair_for_axis_labels(col)
        ax.set_xlabel(x_label, fontsize=18)
        # Left axis = Truth (µT)
        if col == "x":
            left_ylabel = r"$B_x\ [\mu\mathrm{T}]$"
        elif col == "y":
            left_ylabel = r"$B_y\ [\mu\mathrm{T}]$"
        elif col == "z":
            left_ylabel = r"$B_z\ [\mu\mathrm{T}]$"
        elif col == "mag":
            left_ylabel = r"$|B|\ [\mu\mathrm{T}]$"
        else:
            left_ylabel = r"$B\ [\mu\mathrm{T}]$"
        ax.set_ylabel(left_ylabel, fontsize=18)
        ax.tick_params(axis='both', which='major', labelsize=12)

        # Create twin y-axis for truth if requested
        if truth_toggles:
            try:
                ax_t = ax.twinx()
                # Right axis = Forces (N)
                ax_t.set_ylabel(y_label, fontsize=18)
                ax_t.tick_params(axis='both', which='major', labelsize=12)
                axes_truth_map[col] = ax_t
            except Exception:
                axes_truth_map[col] = None

        # Per-load-cell mean±SD bands
        for cell_name, runs in per_cell_runs.items():
            color = cell_to_color.get(cell_name, (0.2, 0.4, 0.8))
            shift = float(cell_shifts.get(cell_name, 0.0))
            for variant, linestyle, alpha in (
                ("sum", "-", 0.18),
                ("inner", "--", 0.14),
                ("outer", ":", 0.12),
            ):
                y_col = _variant_column(col, variant)
                if y_col not in runs[0].columns:
                    continue
                res = _compute_aligned_mean_std(runs, y_col, x_col="frame_aligned")
                if res is None:
                    continue
                grid, means, stds = res
                try:
                    grid = (grid.astype(float) - shift)
                except Exception:
                    pass
                line, = ax.plot(grid, means, color=color, linewidth=2.0, linestyle=linestyle, label=f"{cell_name} {variant}")
                band = ax.fill_between(grid, means - stds, means + stds, color=color, alpha=alpha)
                key = (cell_name, variant)
                grouped_artists.setdefault(key, []).extend([line, band])
                grouped_active.setdefault(key, True)
                if cell_name not in group_to_labels:
                    group_to_labels[cell_name] = []
                if variant not in group_to_labels[cell_name]:
                    group_to_labels[cell_name].append(variant)

                # Apply bands visibility based on global toggle and current line visibility
                try:
                    vis = grouped_active.get(key, True)
                    line.set_visible(vis)
                    band.set_visible(vis and bands_enabled[0])
                except Exception:
                    pass

                # Optional: per-run lines for this variant
                if show_runs:
                    run_label = f"{variant} runs"
                    for df in runs:
                        try:
                            if y_col not in df.columns or "frame_aligned" not in df.columns:
                                continue
                            x = df["frame_aligned"].to_numpy(dtype=float)
                            try:
                                x = x - shift
                            except Exception:
                                pass
                            y = df[y_col].to_numpy(dtype=float)
                            line_run, = ax.plot(x, y, color=color, linewidth=0.9, linestyle="-", alpha=0.35)
                            try:
                                source = df.get("source_path", None)
                                fname = str(source if isinstance(source, str) else (source.iloc[0] if source is not None else ""))
                                line_run._source_path = fname
                            except Exception:
                                try:
                                    line_run._source_path = ""
                                except Exception:
                                    pass
                            grouped_artists.setdefault((cell_name, run_label), []).append(line_run)
                            grouped_active.setdefault((cell_name, run_label), False)
                            try:
                                line_run.set_visible(False)
                            except Exception:
                                pass
                        except Exception:
                            pass
                    if run_label not in group_to_labels[cell_name]:
                        group_to_labels[cell_name].append(run_label)

            # Truth mean and runs (optional)
            if truth_toggles:
                t_col = _truth_column(col)
                if t_col in runs[0].columns:
                    res_t = _compute_aligned_mean_std(runs, t_col, x_col="frame_aligned")
                    if res_t is not None:
                        grid_t, mean_t, std_t = res_t
                        try:
                            grid_t = (grid_t.astype(float) - shift)
                        except Exception:
                            pass
                        line_t, = ax.plot(grid_t, mean_t, color=color, linewidth=2.0, linestyle="-.", label=f"{cell_name} truth")
                        key_t = (cell_name, "truth")
                        grouped_artists.setdefault(key_t, []).append(line_t)
                        grouped_active.setdefault(key_t, True)
                        try:
                            line_t.set_visible(True)
                        except Exception:
                            pass
                        if cell_name not in group_to_labels:
                            group_to_labels[cell_name] = []
                        if "truth" not in group_to_labels[cell_name]:
                            group_to_labels[cell_name].append("truth")

                        if show_runs:
                            run_label_t = "truth runs"
                            for df in runs:
                                try:
                                    if t_col not in df.columns or "frame_aligned" not in df.columns:
                                        continue
                                    xt = df["frame_aligned"].to_numpy(dtype=float)
                                    try:
                                        xt = xt - shift
                                    except Exception:
                                        pass
                                    yt = df[t_col].to_numpy(dtype=float)
                                    line_tr, = ax.plot(xt, yt, color=color, linewidth=0.9, linestyle="-.", alpha=0.35)
                                    try:
                                        source = df.get("source_path", None)
                                        fname = str(source if isinstance(source, str) else (source.iloc[0] if source is not None else ""))
                                        line_tr._source_path = fname
                                    except Exception:
                                        try:
                                            line_tr._source_path = ""
                                        except Exception:
                                            pass
                                    grouped_artists.setdefault((cell_name, run_label_t), []).append(line_tr)
                                    grouped_active.setdefault((cell_name, run_label_t), False)
                                    try:
                                        line_tr.set_visible(False)
                                    except Exception:
                                        pass
                                except Exception:
                                    pass
                            if run_label_t not in group_to_labels[cell_name]:
                                group_to_labels[cell_name].append(run_label_t)

        # Overall across all cells
        for variant, linestyle, alpha in (("sum", "-", 0.20), ("inner", "--", 0.16), ("outer", ":", 0.14)):
            axis_data = overall_data.get(col, {})
            if variant not in axis_data:
                continue
            grid, means, stds = axis_data[variant]
            # Align overall to loaded-cells reference: shift by delta between disk overall ref and loaded ref
            try:
                loaded_ref = _compute_loaded_sumx_median(per_cell_runs, params)
            except Exception:
                loaded_ref = None
            try:
                overall_x = overall_data.get("x", {}).get("sum")
                disk_ref = None
                if overall_x is not None:
                    og, omeans, _ostd = overall_x
                    if og.size and omeans.size:
                        disk_ref = float(og[int(np.nanargmax(omeans.astype(float)))])
            except Exception:
                disk_ref = None
            try:
                if loaded_ref is not None and disk_ref is not None and np.isfinite(loaded_ref) and np.isfinite(disk_ref):
                    delta = float(disk_ref - loaded_ref)
                    grid = (grid.astype(float) - delta)
            except Exception:
                pass
            line, = ax.plot(grid, means, color="#222222", linewidth=2.3, linestyle=linestyle, label=f"overall {variant}")
            band = ax.fill_between(grid, means - stds, means + stds, color="#222222", alpha=alpha)
            key = ("overall", variant)
            grouped_artists.setdefault(key, []).extend([line, band])
            default_visible = (variant == "sum")
            grouped_active.setdefault(key, default_visible)
            try:
                line.set_visible(default_visible)
                band.set_visible(default_visible and bands_enabled[0])
            except Exception:
                pass
            if "overall" not in group_to_labels:
                group_to_labels["overall"] = []
            if variant not in group_to_labels["overall"]:
                group_to_labels["overall"].append(variant)

        # Overall truth (across all cells on disk) on twin axis (if enabled)
        if truth_toggles:
            try:
                t_col = _truth_column(col)
                root = LCPaths().default_root
                cells_disk = _scan_load_cells(root)
                # Reference frame from overall SUM-X
                overall_ref = None
                try:
                    overall_x = overall_data.get("x", {}).get("sum")
                    if overall_x is not None:
                        og, omeans, _ostd = overall_x
                        if og.size and omeans.size:
                            overall_ref = float(og[int(np.nanargmax(omeans.astype(float)))])
                except Exception:
                    overall_ref = None

                grids_truth: List[np.ndarray] = []
                means_truth: List[np.ndarray] = []
                for name in cells_disk.keys():
                    runs_disk = _load_and_process_cell(root / name, params, use_abs_sum=use_abs_sum)
                    if not runs_disk or t_col not in runs_disk[0].columns:
                        continue
                    # per-cell SUM-X peak for shift
                    shift_val = 0.0
                    try:
                        res_sumx = _compute_aligned_mean_std(runs_disk, _variant_column("x", "sum"), x_col="frame_aligned")
                        if res_sumx is not None:
                            gx, mx, _sx = res_sumx
                            if gx.size and mx.size and overall_ref is not None and np.isfinite(overall_ref):
                                idxp = int(np.nanargmax(mx.astype(float)))
                                frame_val = float(gx[idxp]) if 0 <= idxp < gx.size else float(idxp)
                                shift_val = float(frame_val - overall_ref)
                    except Exception:
                        shift_val = 0.0
                    # per-cell truth mean
                    res_t = _compute_aligned_mean_std(runs_disk, t_col, x_col="frame_aligned")
                    if res_t is None:
                        continue
                    grid_t, mean_t, _std_t = res_t
                    try:
                        grid_t = (grid_t.astype(float) - float(shift_val))
                    except Exception:
                        grid_t = grid_t.astype(float)
                    grids_truth.append(grid_t)
                    means_truth.append(mean_t.astype(float))
                if grids_truth:
                    try:
                        gmin = max(float(g.min()) for g in grids_truth)
                        gmax = min(float(g.max()) for g in grids_truth)
                    except Exception:
                        gmin = gmax = None
                    if gmin is not None and gmax is not None and gmax > gmin:
                        grid_common = np.arange(int(np.ceil(gmin)), int(np.floor(gmax)) + 1, dtype=float)
                        stacked = []
                        for grid_t, mean_t in zip(grids_truth, means_truth):
                            idx_map = {int(x): i for i, x in enumerate(grid_t.astype(int))}
                            arr = np.full_like(grid_common, np.nan, dtype=float)
                            for j, xval in enumerate(grid_common.astype(int)):
                                ii = idx_map.get(int(xval))
                                if ii is not None and 0 <= ii < mean_t.size:
                                    arr[j] = float(mean_t[ii])
                            stacked.append(arr)
                        mat = np.vstack(stacked)
                        m_truth = np.nanmean(mat, axis=0)
                        s_truth = np.nanstd(mat, axis=0)
                        ax_truth = axes_truth_map.get(col)
                        target_ax = ax_truth if ax_truth is not None else ax
                        # Shift overall truth to align with loaded reference (same delta)
                        try:
                            if loaded_ref is not None and disk_ref is not None and np.isfinite(loaded_ref) and np.isfinite(disk_ref):
                                delta = float(disk_ref - loaded_ref)
                                grid_common = (grid_common.astype(float) - delta)
                        except Exception:
                            pass
                        line_ot, = target_ax.plot(grid_common, m_truth, color="#222222", linewidth=2.0, linestyle="-.", label=f"overall truth")
                        band_ot = target_ax.fill_between(grid_common, m_truth - s_truth, m_truth + s_truth, color="#222222", alpha=0.12)
                        key_ot = ("overall", "truth")
                        grouped_artists.setdefault(key_ot, []).extend([line_ot, band_ot])
                        grouped_active.setdefault(key_ot, True)
                        try:
                            line_ot.set_visible(True)
                            band_ot.set_visible(bands_enabled[0])
                        except Exception:
                            pass
                        if "overall" not in group_to_labels:
                            group_to_labels["overall"] = []
                        if "truth" not in group_to_labels["overall"]:
                            group_to_labels["overall"].append("truth")
            except Exception:
                pass

        # Group-level across disk (if requested)
        if group_overall:
            for gname in ("CH1-4", "CH5-8", "J"):
                g_axes = group_overall.get(gname, {})
                if not g_axes:
                    continue
                color = {"CH1-4": "#aa3377", "CH5-8": "#33aa77", "J": "#7733aa"}.get(gname, "#555555")
                for variant, linestyle, alpha in (("sum", "-", 0.20), ("inner", "--", 0.16), ("outer", ":", 0.14)):
                    var_key = variant
                    if var_key not in g_axes:
                        continue
                    grid, means, stds = g_axes[var_key]
                    line, = ax.plot(grid, means, color=color, linewidth=2.3, linestyle=linestyle, label=f"{gname} {variant}")
                    band = ax.fill_between(grid, means - stds, means + stds, color=color, alpha=alpha)
                    key = (gname, variant)
                    grouped_artists.setdefault(key, []).extend([line, band])
                    default_visible = (variant == "sum")
                    grouped_active.setdefault(key, default_visible)
                    try:
                        line.set_visible(default_visible)
                        band.set_visible(default_visible and bands_enabled[0])
                    except Exception:
                        pass
                    group_to_labels.setdefault(gname, [])
                    if variant not in group_to_labels[gname]:
                        group_to_labels[gname].append(variant)

    # Right-side controls copied from measured_vs_truth implementation (pagination, remove, hover)
    # Keep user-specified geometry
    try:
        fig.subplots_adjust(top=0.89, bottom=0.085, left=0.065, right=0.805, hspace=0.255, wspace=0.355)
    except Exception:
        pass
    toggle_axes: List = []
    live_widgets: List = []
    live_cursors: List = []
    hover_enabled: List[bool] = [False]
    sample_lines_map: Dict[Tuple[str, str], Line2D] = {}
    # Persist pagination state across UI rebuilds
    pagination_state = {"page": 0}

    def rebuild_toggles() -> None:
        for w in list(live_widgets):
            try:
                if hasattr(w, "disconnect_events"):
                    w.disconnect_events()
            except Exception:
                pass
        live_widgets.clear()
        for ax_t in list(toggle_axes):
            try:
                ax_t.remove()
            except Exception:
                pass
        toggle_axes.clear()
        sample_lines_map.clear()

        if not group_to_labels:
            return
        variant_order = ["sum", "inner", "outer", "truth", "sum runs", "inner runs", "outer runs", "truth runs"]
        for g, lbls in list(group_to_labels.items()):
            ordered = [v for v in variant_order if v in lbls]
            ordered.extend([v for v in lbls if v not in ordered])
            group_to_labels[g] = ordered

        groups_in_order = [g for g in group_to_labels.keys() if g != "overall"]
        if "overall" in group_to_labels:
            groups_in_order.append("overall")

        # Pagination config
        # Show more groups per page when only truth toggles are on
        num_groups = len(groups_in_order)
        if truth_toggles and not show_runs:
            page_size = 3
        elif show_runs:
            page_size = 2
        else:
            page_size = 4
        pagination_enabled = bool(num_groups > page_size)
        total_pages = max(1, (num_groups + page_size - 1) // page_size)
        cur_page = min(max(0, int(pagination_state.get("page", 0))), total_pages - 1)
        pagination_state["page"] = cur_page
        pagination_state["total_pages"] = total_pages
        pagination_state["page_size"] = page_size
        if pagination_enabled:
            start_idx = cur_page * page_size
            end_idx = min(num_groups, start_idx + page_size)
            visible_groups = groups_in_order[start_idx:end_idx]
        else:
            visible_groups = groups_in_order

        x0 = 0.86
        width = 0.12
        y_top = 0.92
        y_bottom = 0.12
        available = max(0.0, y_top - y_bottom)
        title_h = 0.02
        gap_h = 0.02
        base_item_h = 0.055
        min_item_h = 0.035
        total_items = sum(len(group_to_labels[g]) for g in visible_groups)
        content_h_base = len(visible_groups) * (title_h + gap_h) + total_items * base_item_h
        item_h = base_item_h if content_h_base <= available else min(base_item_h, max(min_item_h, (available - len(visible_groups) * (title_h + gap_h)) / max(1, total_items)))

        chks: List[CheckButtons] = []
        style_map = {"sum": "-", "inner": "--", "outer": ":", "truth": "-."}
        y_cursor = y_top
        for group_name in visible_groups:
            labels = group_to_labels.get(group_name, [])
            n_items = len(labels)
            if n_items == 0:
                continue
            try:
                title_offset = 0.008
                title_y = y_cursor - title_offset
                title_visible = (title_y <= (y_top + 0.01)) and (title_y >= (y_bottom - 0.03))
                if title_visible:
                    title_ax = fig.add_axes([x0, title_y, width - 0.03, 0.02])
                    title_ax.axis('off')
                    title_ax.text(0, 0, group_name, fontsize=10, fontweight="bold", ha="left", va="bottom")
                    toggle_axes.append(title_ax)
                if group_name != "overall" and title_visible:
                    rm_ax = fig.add_axes([x0 + width - 0.028, title_y, 0.024, 0.02])
                    rm_btn = Button(rm_ax, "x")
                    def make_on_remove(gn: str):
                        def _on_remove(_event) -> None:
                            main_axes = [ax for ax in getattr(axes, "ravel", lambda: [])()] if hasattr(axes, "ravel") else list(axes)
                            for variant in group_to_labels.get(gn, []):
                                arts = grouped_artists.get((gn, variant), [])
                                new_keep = []
                                for a in arts:
                                    try:
                                        if getattr(a, "axes", None) in main_axes:
                                            a.remove()
                                        else:
                                            new_keep.append(a)
                                    except Exception:
                                        new_keep.append(a)
                                grouped_artists[(gn, variant)] = new_keep
                                grouped_active.pop((gn, variant), None)
                            group_to_labels.pop(gn, None)
                            per_cell_runs.pop(gn, None)
                            rebuild_toggles()
                            try:
                                fig.canvas.draw_idle()
                            except Exception:
                                pass
                        return _on_remove
                    rm_btn.on_clicked(make_on_remove(group_name))
                    toggle_axes.append(rm_ax)
                    live_widgets.append(rm_btn)
            except Exception:
                pass

            height = n_items * item_h
            y0 = y_cursor - title_h - height
            togg_ax = fig.add_axes([x0, y0, width, height])
            actives = [grouped_active.get((group_name, v), True) for v in labels]
            chk = CheckButtons(togg_ax, labels=labels, actives=actives)
            chks.append(chk)
            toggle_axes.append(togg_ax)
            live_widgets.append(chk)

            try:
                for txt in chk.labels:
                    txt.set_fontsize(9)
            except Exception:
                pass

            try:
                for txt, v in zip(chk.labels, labels):
                    trans = txt.get_transform()
                    x_txt, y_txt = txt.get_position()
                    is_runs = (" runs" in v)
                    base_x1 = min(0.80, x_txt + 0.20)
                    if is_runs:
                        x1 = min(0.94, base_x1 + 0.08)
                    else:
                        x1 = base_x1
                    x2 = min(0.98, x1 + 0.16)
                    if group_name == "overall":
                        color = "#222222"
                    else:
                        group_color_map = {"CH1-4": "#aa3377", "CH5-8": "#33aa77", "J": "#7733aa"}
                        color = group_color_map.get(group_name, cell_to_color.get(group_name, (0.2, 0.4, 0.8)))
                    base_v = v.split(" ")[0] if " " in v else v
                    linestyle = style_map.get(base_v, "-")
                    line = plt.Line2D([x1, x2], [y_txt, y_txt], color=color, linestyle=linestyle, linewidth=2.6, solid_capstyle="round")
                    line.set_transform(trans)
                    line.set_zorder(2000)
                    line.set_clip_on(False)
                    line.set_visible(grouped_active.get((group_name, v), True))
                    togg_ax.add_line(line)
                    sample_lines_map[(group_name, v)] = line
            except Exception:
                pass

            def _make_cb(gn: str, labels_ref: List[str], chk_ref: CheckButtons):
                def on_toggle(label: str) -> None:
                    try:
                        idx = labels_ref.index(label)
                    except ValueError:
                        return
                    visible = chk_ref.get_status()[idx]
                    grouped_active[(gn, label)] = visible
                    # Apply visibility: lines follow "visible"; bands follow "visible and bands_enabled"
                    for a in grouped_artists.get((gn, label), []):
                        try:
                            if isinstance(a, Line2D):
                                a.set_visible(visible)
                            else:
                                a.set_visible(visible and bands_enabled[0])
                        except Exception:
                            pass
                    try:
                        sl = sample_lines_map.get((gn, label))
                        if sl is not None:
                            sl.set_visible(visible)
                    except Exception:
                        pass
                    try:
                        fig.canvas.draw_idle()
                    except Exception:
                        pass
                return on_toggle

            chk.on_clicked(_make_cb(group_name, labels, chk))
            y_cursor = y0 - gap_h

        # Enforce global band visibility after rebuild
        try:
            # Build axes set including twin truth axes
            main_axes = [ax for ax in getattr(axes, "ravel", lambda: [])()] if hasattr(axes, "ravel") else list(axes)
            truth_axes = []
            try:
                truth_axes = [ax_t for ax_t in axes_truth_map.values() if ax_t is not None]
            except Exception:
                truth_axes = []
            axes_set = set(main_axes + truth_axes)
            for key, arts in grouped_artists.items():
                # Determine if the primary line for this key is currently visible
                line_visible = False
                for a in arts:
                    if isinstance(a, Line2D) and getattr(a, "axes", None) in axes_set and a.get_visible():
                        line_visible = True
                        break
                for a in arts:
                    if getattr(a, "axes", None) in axes_set and not isinstance(a, Line2D):
                        try:
                            a.set_visible(bands_enabled[0] and line_visible)
                        except Exception:
                            pass
        except Exception:
            pass

        # All OFF button (compact)
        try:
            btn_w = min(width, 0.10)
            btn_h = 0.03
            y_all_off = 0.075
            x_all_off = x0 + max(0.0, (width - btn_w) * 0.5)
            all_off_ax = fig.add_axes([x_all_off, y_all_off, btn_w, btn_h])
            all_off_btn = Button(all_off_ax, "All OFF")
            toggle_axes.append(all_off_ax)
            live_widgets.append(all_off_btn)

            def on_all_off(_event) -> None:
                try:
                    for gn, lbls in list(group_to_labels.items()):
                        for v in list(lbls):
                            grouped_active[(gn, v)] = False
                            for a in grouped_artists.get((gn, v), []):
                                try:
                                    a.set_visible(False)
                                except Exception:
                                    pass
                            try:
                                sl = sample_lines_map.get((gn, v))
                                if sl is not None:
                                    sl.set_visible(False)
                            except Exception:
                                pass
                    rebuild_toggles()
                    try:
                        fig.canvas.draw_idle()
                    except Exception:
                        pass
                except Exception:
                    pass

            all_off_btn.on_clicked(on_all_off)
        except Exception:
            pass

        # Pagination controls
        try:
            if num_groups > page_size:
                y_btn = 0.03
                btn_h = 0.03
                prev_w = 0.04
                next_w = 0.04
                prev_ax = fig.add_axes([x0, y_btn, prev_w, btn_h])
                next_ax = fig.add_axes([x0 + width - next_w, y_btn, next_w, btn_h])
                prev_btn = Button(prev_ax, "Prev")
                next_btn = Button(next_ax, "Next")
                toggle_axes.extend([prev_ax, next_ax])
                live_widgets.extend([prev_btn, next_btn])

                mid_h = 0.015
                mid_y = y_btn + max(0.0, (btn_h - mid_h) * 0.5)
                mid_x = x0 + prev_w + 0.004
                mid_w = max(0.0, width - (prev_w + next_w + 0.008))
                mid_ax = fig.add_axes([mid_x, mid_y, mid_w, mid_h])
                mid_ax.axis('off')
                try:
                    mid_ax.text(0.5, 0.0, f"Page {cur_page + 1}/{total_pages}", fontsize=8, ha="center", va="bottom")
                except Exception:
                    pass
                toggle_axes.append(mid_ax)

                def _on_prev(_event) -> None:
                    try:
                        current = int(pagination_state.get("page", 0))
                        if current > 0:
                            pagination_state["page"] = current - 1
                        rebuild_toggles()
                        try:
                            fig.canvas.draw_idle()
                        except Exception:
                            pass
                    except Exception:
                        pass

                def _on_next(_event) -> None:
                    try:
                        current = int(pagination_state.get("page", 0))
                        if current < total_pages - 1:
                            pagination_state["page"] = current + 1
                        rebuild_toggles()
                        try:
                            fig.canvas.draw_idle()
                        except Exception:
                            pass
                    except Exception:
                        pass

                prev_btn.on_clicked(_on_prev)
                next_btn.on_clicked(_on_next)
        except Exception:
            pass

    rebuild_toggles()

    # Function to recompute overall from disk and refresh only overall artists
    def recompute_overall_and_refresh() -> None:
        try:
            new_overall = _compute_overall_cellwise_from_disk(params, use_abs_sum=use_abs_sum)
            main_axes = [ax for ax in getattr(axes, "ravel", lambda: [])()] if hasattr(axes, "ravel") else list(axes)
            # Remove previous overall artists from main axes, keep sample lines
            for variant in ("sum", "inner", "outer"):
                key = ("overall", variant)
                prev = grouped_artists.get(key, [])
                kept = []
                for a in prev:
                    try:
                        if getattr(a, "axes", None) in main_axes:
                            a.remove()
                        else:
                            kept.append(a)
                    except Exception:
                        kept.append(a)
                grouped_artists[key] = kept

            # Re-add refreshed overall lines
            for col in ["x", "y", "z", "mag"]:
                r, c = {"x": (0, 0), "y": (0, 1), "z": (1, 0), "mag": (1, 1)}[col]
                ax = axes[r][c]
                axis_data = new_overall.get(col, {})
                for variant, linestyle, alpha in (("sum", "-", 0.20), ("inner", "--", 0.16), ("outer", ":", 0.14)):
                    if variant not in axis_data:
                        continue
                    grid, means, stds = axis_data[variant]
                    line, = ax.plot(grid, means, color="#222222", linewidth=2.3, linestyle=linestyle, label=f"overall {variant}")
                    band = ax.fill_between(grid, means - stds, means + stds, color="#222222", alpha=alpha)
                    key = ("overall", variant)
                    grouped_artists.setdefault(key, []).extend([line, band])
                    visible = grouped_active.get(key, variant == "sum")
                    try:
                        line.set_visible(visible)
                        band.set_visible(visible)
                    except Exception:
                        pass
            try:
                fig.canvas.draw_idle()
            except Exception:
                pass
        except Exception:
            pass

    # Hover labels for per-cell lines
    try:
        manager = getattr(fig.canvas, "manager", None)
        toolbar = getattr(manager, "toolbar", None)
        if toolbar is not None and mplcursors is not None:
            try:
                toolbar.addSeparator()
            except Exception:
                pass
            hover_action = toolbar.addAction("Hover Labels (OFF)")
            bands_action = toolbar.addAction("Bands (ON)")
            try:
                hover_action.setCheckable(True)
                hover_action.setChecked(False)
                bands_action.setCheckable(True)
                bands_action.setChecked(True)
            except Exception:
                pass
            # Only hover individual run lines (like measured_vs_truth): use _source_path on artists

            def _get_visible_lines() -> List[Line2D]:
                lines: List[Line2D] = []
                main_axes = [ax for ax in getattr(axes, "ravel", lambda: [])()] if hasattr(axes, "ravel") else list(axes)
                for (group_name, variant), arts in grouped_artists.items():
                    if group_name == "overall":
                        continue
                    if not isinstance(variant, str) or not variant.endswith(" runs"):
                        continue
                    for a in arts:
                        if isinstance(a, Line2D) and getattr(a, "axes", None) in main_axes and a.get_visible():
                            lines.append(a)
                return lines

            live_cursors_local: List = []

            def _apply_cursors():
                for cur in list(live_cursors_local):
                    try:
                        cur.remove()
                    except Exception:
                        pass
                live_cursors_local.clear()
                if not hover_enabled[0]:
                    return
                artists = _get_visible_lines()
                if not artists:
                    return
                try:
                    cur = mplcursors.cursor(artists, hover=True)
                    @cur.connect("add")
                    def on_add(sel):
                        try:
                            artist = sel.artist
                            fname = getattr(artist, "_source_path", None)
                            sel.annotation.set_text(str(fname or ""))
                        except Exception:
                            pass
                    live_cursors_local.append(cur)
                except Exception:
                    pass

            def _on_toggle_hover(checked: bool=False):
                hover_enabled[0] = bool(checked)
                try:
                    hover_action.setText("Hover Labels (ON)" if hover_enabled[0] else "Hover Labels (OFF)")
                except Exception:
                    pass
                _apply_cursors()
                try:
                    fig.canvas.draw_idle()
                except Exception:
                    pass

            hover_action.triggered.connect(_on_toggle_hover)
            def _on_toggle_bands(checked: bool=False):
                bands_enabled[0] = bool(checked)
                try:
                    bands_action.setText("Bands (ON)" if bands_enabled[0] else "Bands (OFF)")
                except Exception:
                    pass
                # Toggle bands only for groups whose lines are currently visible in the main/twin axes
                try:
                    main_axes = [ax for ax in getattr(axes, "ravel", lambda: [])()] if hasattr(axes, "ravel") else list(axes)
                    truth_axes = []
                    try:
                        truth_axes = [ax_t for ax_t in axes_truth_map.values() if ax_t is not None]
                    except Exception:
                        truth_axes = []
                    axes_set = set(main_axes + truth_axes)
                    for key, arts in grouped_artists.items():
                        # Is the corresponding line visible?
                        line_visible = any(isinstance(a, Line2D) and getattr(a, "axes", None) in axes_set and a.get_visible() for a in arts)
                        for a in arts:
                            if getattr(a, "axes", None) in axes_set and not isinstance(a, Line2D):
                                try:
                                    a.set_visible(bands_enabled[0] and line_visible)
                                except Exception:
                                    pass
                    fig.canvas.draw_idle()
                except Exception:
                    pass

            try:
                bands_action.triggered.connect(_on_toggle_bands)
            except Exception:
                pass
            try:
                fig.canvas.mpl_connect('draw_event', lambda _evt: _apply_cursors())
            except Exception:
                pass
    except Exception:
        pass

    # Attach toolbar actions (Qt backends only)
    _attach_add_runs_to_project_button(
        fig=fig,
        axes=axes,
        params=params,
        recompute_overall_and_refresh=recompute_overall_and_refresh,
    )

    _attach_add_cells_to_plot_button(
        fig=fig,
        axes=axes,
        per_cell_runs=per_cell_runs,
        params=params,
        grouped_artists=grouped_artists,
        grouped_active=grouped_active,
        group_to_labels=group_to_labels,
        cell_to_color=cell_to_color,
        palette=palette,
        rebuild_toggles=rebuild_toggles,
        use_abs_sum=use_abs_sum,
        show_runs=show_runs,
        truth_toggles=truth_toggles,
        axes_truth_map=axes_truth_map,
        bands_enabled=bands_enabled,
    )

    suptitle = "Measured vs Time/Frames by Load Cell"
    if title_suffix:
        suptitle = f"{suptitle} — {title_suffix}"
    fig.suptitle(suptitle, fontsize=20, y=0.95)

    if save_path is not None:
        try:
            save_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(str(save_path), dpi=150, bbox_inches="tight")
        except Exception:
            pass

    if show:
        plt.show()
    else:
        plt.close(fig)


def main() -> None:
    lc_paths = LCPaths()
    parser = argparse.ArgumentParser(description="Plot measured vs time/frames per load cell (mounted only), with mean±SD bands for sum/inner/outer.")
    parser.add_argument("--root", type=str, default=str(lc_paths.default_root), help="Root folder containing load cell folders (each with *.csv)")
    parser.add_argument("--auto-source", type=str, default=str(lc_paths.auto_source_root), help="Root folder to recursively scan for new CIR *.csv files to auto-populate Load_Cell_Runs")
    parser.add_argument("--load-cell", type=str, default=None, help="Single load cell folder name to plot (subdir under --root)")
    parser.add_argument("--save", action="store_true", help="Save figure to outputs/plots")
    parser.add_argument("--no-show", action="store_false", dest="show", default=True, help="Do not show figure interactively")
    parser.add_argument("--abs-sum", action="store_true", dest="use_abs_sum", help="Compute 'sum' as |inner|+|outer| per axis instead of using 'sum' columns")
    parser.add_argument("--show-runs", action="store_true", dest="show_runs", help="Enable per-run line toggles (sum/inner/outer) and hover with file name")
    parser.add_argument("--truth-toggles", action="store_true", dest="truth_toggles", help="Enable truth mean toggles and, with --show-runs, truth runs; limit pagination to one cell per page")
    parser.add_argument("--add-groups", action="store_true", dest="add_groups", help="Add group-level overall lines (CH1-4, CH5-8, J) with mean±SD for sum/inner/outer")
    args = parser.parse_args()

    root = Path(args.root)
    auto_src = Path(args.auto_source)

    # Auto-populate Load_Cell_Runs from Load_Cell_Spiral_test by scanning for CIR CSVs
    try:
        if auto_src.exists():
            for csv_path in auto_src.rglob("*.csv"):
                name_lower = csv_path.name.lower()
                if "cir" not in name_lower:
                    continue
                parts = csv_path.name.split("_")
                if len(parts) < 2:
                    continue
                load_cell_name = parts[-2].strip()
                if not load_cell_name:
                    continue
                dest_dir = root / load_cell_name
                dest_dir.mkdir(parents=True, exist_ok=True)
                dest_file = dest_dir / csv_path.name
                try:
                    if not dest_file.exists():
                        shutil.copy2(str(csv_path), str(dest_file))
                except Exception:
                    pass
    except Exception:
        pass

    params = Params()
    try:
        # Default this script to align by smoothed +X peak
        params.alignment_method = "x_peak_smoothed"
    except Exception:
        pass
    per_cell_runs: Dict[str, List[pd.DataFrame]] = {}
    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    save_path: Optional[Path] = None
    if args.save:
        save_dir = lc_paths.outputs_root
        save_path = save_dir / f"load_cells_time_series_{timestamp}.png"

    title_suffix = args.load_cell if args.load_cell else ""
    plot_load_cells_time_series(
        per_cell_runs=per_cell_runs,
        params=params,
        title_suffix=title_suffix,
        save_path=save_path,
        show=bool(args.show),
        use_abs_sum=bool(getattr(args, "use_abs_sum", False)),
        show_runs=bool(getattr(args, "show_runs", False)),
        truth_toggles=bool(getattr(args, "truth_toggles", False)),
        add_groups=bool(getattr(args, "add_groups", False)),
    )


if __name__ == "__main__":
    main()


