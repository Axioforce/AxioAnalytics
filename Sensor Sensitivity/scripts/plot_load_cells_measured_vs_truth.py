import os
import glob
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
import re
import logging

# Logger for this module
logger = logging.getLogger("sensor_sensitivity.load_cells_measured_vs_truth")
if not logger.handlers:
    try:
        logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    except Exception:
        pass

try:
    # Use Matplotlib's Qt compatibility layer to support PyQt5/PySide2/PyQt6/PySide6
    from matplotlib.backends.qt_compat import QtWidgets  # type: ignore
except Exception:  # pragma: no cover - safe fallback when non-Qt backend
    QtWidgets = None

try:
    import mplcursors  # type: ignore
except Exception:
    mplcursors = None

# Reuse pipeline helpers and configuration from the overlay module
from overlay_mounted_vs_reseated import (
    Params,
    load_group,
    preprocess_signal,
    rotate_group_about_z,
    align_group_by_peak,
    align_group_by_local_peak,
    align_run,
    compute_group_stats_aligned,
    smooth_for_alignment,
)


def _pair_for_axis(axis: str) -> Tuple[str, str]:
    """Return (measured_col, truth_col) mapping for an axis.

    x->(x,bx), y->(y,by), z->(z,bz), mag->(mag,bmag)
    """
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
    """Return (measured_label, truth_label) with units and formatting, matching existing scripts."""
    if axis == "x":
        return "$F_x$ [N]", r"$B_x\ [\mu\mathrm{T}]$"
    if axis == "y":
        return "$F_y$ [N]", r"$B_y\ [\mu\mathrm{T}]$"
    if axis == "z":
        return "$F_z$ [N]", r"$B_z\ [\mu\mathrm{T}]$"
    if axis == "mag":
        return "$|F|$ [N]", r"$|B|\ [\mu\mathrm{T}]$"
    return axis, axis


def _variant_column(base: str, variant: str) -> str:
    """Return column name for a given base axis and variant (sum|inner|outer)."""
    if variant == "sum":
        return base if base != "mag" else "mag"
    if variant == "inner":
        return f"{base}_inner" if base != "mag" else "mag_inner"
    if variant == "outer":
        return f"{base}_outer" if base != "mag" else "mag_outer"
    return base


def _compute_binned_mean_std(
    runs: List[pd.DataFrame],
    meas_col: str,
    truth_col: str,
    n_bins: int = 60,
) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Compute binned mean±std of measured vs truth across all runs.

    Returns (centers, means, stds) or None if insufficient data.
    """
    try:
        xs: List[np.ndarray] = []
        ys: List[np.ndarray] = []
        tx_min: Optional[float] = None
        tx_max: Optional[float] = None
        for df in runs:
            if meas_col not in df.columns or truth_col not in df.columns:
                continue
            x_truth = df[truth_col].to_numpy(dtype=float)
            y_meas = df[meas_col].to_numpy(dtype=float)
            n = min(x_truth.size, y_meas.size)
            if n <= 0:
                continue
            xs.append(x_truth[:n])
            ys.append(y_meas[:n])
            cur_min = float(np.nanmin(x_truth[:n])) if n > 0 else None
            cur_max = float(np.nanmax(x_truth[:n])) if n > 0 else None
            if cur_min is not None:
                tx_min = cur_min if tx_min is None else min(tx_min, cur_min)
            if cur_max is not None:
                tx_max = cur_max if tx_max is None else max(tx_max, cur_max)

        if not xs or tx_min is None or tx_max is None or not np.isfinite(tx_min) or not np.isfinite(tx_max):
            return None
        if tx_min == tx_max:
            tx_max = tx_min + 1.0

        edges = np.linspace(tx_min, tx_max, int(n_bins) + 1)
        centers = 0.5 * (edges[:-1] + edges[1:])

        all_x = np.concatenate(xs).astype(float)
        all_y = np.concatenate(ys).astype(float)
        means = np.full_like(centers, np.nan, dtype=float)
        stds = np.full_like(centers, np.nan, dtype=float)
        for i in range(centers.size):
            mask = (all_x >= edges[i]) & (all_x < edges[i + 1])
            vals = all_y[mask]
            if vals.size:
                means[i] = float(np.nanmean(vals))
                stds[i] = float(np.nanstd(vals))
        return centers, means, stds
    except Exception:
        return None


def _trim_by_truth_z_slope(
    runs: List[pd.DataFrame],
    slope_threshold: float = -1.0,
    smooth_window: int = 10,
) -> List[pd.DataFrame]:
    """Trim runs by first index where mean bz slope falls below threshold.

    Uses aligned grid if available via 'frame_aligned'. If none, returns input.
    """
    try:
        combined: List[pd.DataFrame] = []
        for df in runs:
            if df is not None and ("frame_aligned" in df.columns) and ("bz" in df.columns):
                combined.append(df[["frame_aligned", "bz"]].copy())
        if not combined:
            return runs
        stats, grid = compute_group_stats_aligned(combined, ["bz"], x_col="frame_aligned")
        if not stats or grid.size == 0 or "bz" not in stats:
            return runs
        z_mean = stats["bz"]["mean"].to_numpy()
        z_smooth = smooth_for_alignment(z_mean, smooth_window)
        slopes = np.diff(z_smooth)
        idx: Optional[int] = None
        for i in range(slopes.size):
            if np.isfinite(slopes[i]) and slopes[i] <= slope_threshold:
                idx = i + 1
                break
        if idx is None:
            return runs
        cutoff_frame = int(grid[idx])

        out: List[pd.DataFrame] = []
        for d in runs:
            if "frame_aligned" in d.columns:
                out.append(d.loc[d["frame_aligned"] <= cutoff_frame].reset_index(drop=True).copy())
            else:
                out.append(d)
        return out
    except Exception:
        return runs


@dataclass
class LCPaths:
    project_root: Path = Path(__file__).resolve().parent.parent
    default_root: Path = project_root / "Load_Cell_Runs"
    outputs_root: Path = project_root / "outputs" / "plots" / "load_cells_measured_vs_truth"
    auto_source_root: Path = project_root / "Load_Cell_Spiral_test"


def _scan_load_cells(root: Path, only_cell: Optional[str] = None) -> Dict[str, str]:
    """Return map of load_cell_name -> glob for runs CSVs (per load cell)."""
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


def _attach_add_runs_to_project_button(
    fig,
    axes,
    per_cell_runs: Dict[str, List[pd.DataFrame]],
    params: Params,
    grouped_artists: Dict[Tuple[str, str], List],
    grouped_active: Dict[Tuple[str, str], bool],
    recompute_overall_and_refresh,
) -> None:
    """Attach a Qt toolbar button "Add Runs to Project" to select CSVs, copy into
    Load_Cell_Runs/<name>, reload that cell into memory if desired, and refresh overall lines.

    No-ops gracefully if not running under a Qt backend.
    """
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

        def _on_add_load_cell() -> None:
            try:
                # 1) Select one or more CSV files
                start_dir = str(LCPaths().default_root)
                files, _ = QtWidgets.QFileDialog.getOpenFileNames(
                    None,
                    "Select CSV files",
                    start_dir,
                    "CSV Files (*.csv)"
                )
                if not files:
                    return

                # 2) Prompt for a load cell name
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

                # 3) Create destination folder and copy files
                dest_dir = LCPaths().default_root / name
                dest_dir.mkdir(parents=True, exist_ok=True)
                for src in files:
                    try:
                        dst = dest_dir / os.path.basename(src)
                        shutil.copy2(src, dst)
                    except Exception:
                        # Continue copying other files even if one fails
                        pass

                # 4) Recompute and refresh overall lines from disk
                recompute_overall_and_refresh()

            except Exception:
                # Swallow to avoid killing the UI if anything goes wrong
                pass

        action.triggered.connect(_on_add_load_cell)
    except Exception:
        # If anything fails, do not crash the plotting
        pass


def _load_and_process_cell(cell_dir: Path, params: Params) -> List[pd.DataFrame]:
    try:
        dfs = load_group(str(cell_dir / "*.csv"), params.axis_mapping)
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
        dfs_trim = _trim_by_truth_z_slope(dfs_rot)
        return dfs_trim
    except Exception:
        return []


def _compute_overall_cellwise_from_disk(params: Params, n_bins: int = 60) -> Dict[str, Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]]]:
    """Compute overall lines per axis and variant by averaging per-cell means.

    Returns mapping: axis -> variant -> (centers, mean_across_cells, std_across_cells)
    """
    root = LCPaths().default_root
    cells = _scan_load_cells(root)
    if not cells:
        return {}

    result: Dict[str, Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]]] = {}
    axes_list = ["x", "y", "z", "mag"]

    # Preload processed runs per cell (once)
    cell_to_runs: Dict[str, List[pd.DataFrame]] = {}
    for name in cells.keys():
        runs = _load_and_process_cell(root / name, params)
        if runs:
            cell_to_runs[name] = runs

    for axis in axes_list:
        meas_col, truth_col = _pair_for_axis(axis)
        # Determine global bin edges from all runs of all cells
        tx_min: Optional[float] = None
        tx_max: Optional[float] = None
        for runs in cell_to_runs.values():
            for df in runs:
                if truth_col not in df.columns:
                    continue
                arr = df[truth_col].to_numpy(dtype=float)
                if arr.size == 0:
                    continue
                cur_min = float(np.nanmin(arr))
                cur_max = float(np.nanmax(arr))
                tx_min = cur_min if tx_min is None else min(tx_min, cur_min)
                tx_max = cur_max if tx_max is None else max(tx_max, cur_max)
        if tx_min is None or tx_max is None:
            continue
        if tx_min == tx_max:
            tx_max = tx_min + 1.0
        edges = np.linspace(tx_min, tx_max, int(n_bins) + 1)
        centers = 0.5 * (edges[:-1] + edges[1:])

        axis_results: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
        for variant in ("sum", "inner", "outer"):
            v_col = _variant_column(meas_col, variant)
            # Collect per-cell mean arrays
            per_cell_means: List[np.ndarray] = []
            for runs in cell_to_runs.values():
                if not runs or v_col not in runs[0].columns:
                    continue
                # Concatenate all runs for this cell and compute mean per bin
                try:
                    all_x = []
                    all_y = []
                    for df in runs:
                        if truth_col in df.columns and v_col in df.columns:
                            x = df[truth_col].to_numpy(dtype=float)
                            y = df[v_col].to_numpy(dtype=float)
                            n = min(x.size, y.size)
                            if n > 0:
                                all_x.append(x[:n])
                                all_y.append(y[:n])
                    if not all_x:
                        continue
                    all_x_arr = np.concatenate(all_x).astype(float)
                    all_y_arr = np.concatenate(all_y).astype(float)
                    means = np.full_like(centers, np.nan, dtype=float)
                    for i in range(centers.size):
                        mask = (all_x_arr >= edges[i]) & (all_x_arr < edges[i + 1])
                        vals = all_y_arr[mask]
                        if vals.size:
                            means[i] = float(np.nanmean(vals))
                    per_cell_means.append(means)
                except Exception:
                    continue
            if not per_cell_means:
                continue
            cell_matrix = np.vstack(per_cell_means)
            mean_across_cells = np.nanmean(cell_matrix, axis=0)
            std_across_cells = np.nanstd(cell_matrix, axis=0)
            axis_results[variant] = (centers, mean_across_cells, std_across_cells)
        if axis_results:
            result[axis] = axis_results

    return result


def _device_master_list_path() -> Path:
    try:
        # Project root points to "Sensor Sensitivity" directory
        p = LCPaths().project_root / "Device_Master_List" / "Device Master List.xlsx"
        try:
            logger.info(f"Device Master List path resolved: {p} (exists={p.exists()})")
        except Exception:
            pass
        return p
    except Exception:
        return Path("Device Master List.xlsx")


@lru_cache(maxsize=1)
def _load_magnets_diag_table() -> Dict[str, Tuple[float, float, float, float]]:
    """Return mapping of diag-id -> (outer_x, outer_y, inner_x, inner_y).

    Reads the Magnets sheet (rows 120-124, cols B-G) as specified by the user.
    """
    out: Dict[str, Tuple[float, float, float, float]] = {}
    try:
        xlsx = _device_master_list_path()
        # Resolve magnets sheet name robustly, then read range
        magnets_sheet = _resolve_sheet_name(
            xlsx_path=xlsx,
            primary_targets=["magnets"],
            fallback_tokens=["magnet"],
        )
        logger.info("Loading Magnets sheet B120:G124 for offsets ...")
        # Read exact range B120:G124 (5 rows) with pandas; header=None preserves raw layout
        tbl = pd.read_excel(
            xlsx,
            sheet_name=magnets_sheet if magnets_sheet else "Magnets",
            header=None,
            usecols="B:G",
            skiprows=119,  # zero-based, so 119 -> starts at Excel row 120
            nrows=5,
            engine=None,  # let pandas auto-select (openpyxl)
        )
        try:
            logger.info(f"Magnets table loaded: shape={tbl.shape}")
        except Exception:
            pass
        # Expected columns: [Diag, outer_x, outer_y, (empty), inner_x, inner_y]
        for _, row in tbl.iterrows():
            try:
                key_raw = str(row.iloc[0]).strip()
                if not key_raw or key_raw.lower() == "nan":
                    continue
                def _to_f(v) -> float:
                    try:
                        return float(v)
                    except Exception:
                        return float("nan")
                outer_x = _to_f(row.iloc[1]) if row.size > 1 else float("nan")
                outer_y = _to_f(row.iloc[2]) if row.size > 2 else float("nan")
                # There's an empty column between outer_y and inner_x; shift inner indices by +1
                inner_x = _to_f(row.iloc[4]) if row.size > 4 else float("nan")
                inner_y = _to_f(row.iloc[5]) if row.size > 5 else float("nan")
                out[key_raw] = (outer_x, outer_y, inner_x, inner_y)
            except Exception:
                continue
        try:
            logger.info(f"Magnets diag keys parsed: {list(out.keys())}")
        except Exception:
            pass
    except Exception as e:
        try:
            logger.warning(f"Failed to load magnets table: {e}")
        except Exception:
            pass
        return {}
    return out


def _normalize_diag_key(s: str) -> str:
    """Normalize diag identifiers for comparison: lowercase, remove spaces and commas.
    Example: 'Diag 5 A' -> 'diag5a', 'Diag5, a-type' -> 'diag5a'.
    """
    s2 = re.sub(r"[,\s]+", "", str(s)).lower()
    return s2


def _normalize_sheet_name(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(s).lower())


def _resolve_sheet_name(xlsx_path: Path, primary_targets: List[str], fallback_tokens: Optional[List[str]] = None) -> Optional[str]:
    """Return best sheet name match given target names (fuzzy, case/space-insensitive).

    - Tries exact normalized match against each name in primary_targets
    - Then tries sheets containing all fallback_tokens (normalized)
    - Then tries sheets containing any token
    Logs available sheets and selection.
    """
    try:
        xls = pd.ExcelFile(xlsx_path)
    except Exception as e:
        try:
            logger.warning(f"Failed to open Excel file '{xlsx_path}': {e}")
        except Exception:
            pass
        return None
    sheets = list(xls.sheet_names)
    try:
        logger.info(f"Available sheets: {sheets}")
    except Exception:
        pass
    norm_map = {sheet: _normalize_sheet_name(sheet) for sheet in sheets}
    # 1) exact normalized match
    targets_norm = [_normalize_sheet_name(t) for t in primary_targets]
    for sheet, norm in norm_map.items():
        if norm in targets_norm:
            try:
                logger.info(f"Selected sheet by exact match: {sheet}")
            except Exception:
                pass
            return sheet
    # 2) contains all fallback tokens
    if fallback_tokens:
        fb_norm = [_normalize_sheet_name(t) for t in fallback_tokens]
        for sheet, norm in norm_map.items():
            if all(tok in norm for tok in fb_norm):
                try:
                    logger.info(f"Selected sheet by all-tokens match: {sheet}")
                except Exception:
                    pass
                return sheet
        # 3) contains any token
        for sheet, norm in norm_map.items():
            if any(tok in norm for tok in fb_norm):
                try:
                    logger.info(f"Selected sheet by any-token match: {sheet}")
                except Exception:
                    pass
                return sheet
    try:
        logger.warning("No matching sheet found with provided targets/tokens")
    except Exception:
        pass
    return None


def _parse_diag_from_notes(notes: str) -> Optional[Tuple[int, Optional[str]]]:
    """Extract (number, optional letter) from inconsistent Notes strings.

    Recognizes forms like:
    - 'Diag4'
    - 'Diag 5 A' / 'Diag 5A'
    - 'Diag 5, a-type(top MB)' / 'Diag 5, b-type(bottom MB)'
    - 'Diag 5, new ...' (returns number only)
    """
    if not notes:
        return None
    text = str(notes)
    try:
        logger.info(f"Parsing notes for diag: '{text}'")
    except Exception:
        pass
    m = re.search(r"(?i)diag\s*(\d+)\s*([ab])\b", text)
    if m:
        num = int(m.group(1))
        letter = m.group(2).upper()
        try:
            logger.info(f"Parsed diag -> number={num}, letter={letter}")
        except Exception:
            pass
        return num, letter
    m = re.search(r"(?i)diag\s*(\d+)", text)
    if not m:
        return None
    num = int(m.group(1))
    letter = None
    # infer letter from 'a-type' / 'b-type' tokens
    if re.search(r"(?i)\ba[-\s]?type\b", text):
        letter = "A"
    elif re.search(r"(?i)\bb[-\s]?type\b", text):
        letter = "B"
    try:
        logger.info(f"Parsed diag -> number={num}, letter={letter}")
    except Exception:
        pass
    return num, letter


def _canonical_diag_label(num: int, letter: Optional[str]) -> str:
    return f"Diag {num}{letter}" if letter else f"Diag {num}"


def _find_offsets_by_diag_identifier(diag_text: str) -> Optional[Tuple[str, Tuple[float, float, float, float]]]:
    """Match inconsistent diag text to magnets table entry.

    Returns (matched_key, offsets) if found.
    """
    magnets = _load_magnets_diag_table()
    if not magnets:
        return None
    parsed = _parse_diag_from_notes(diag_text)
    candidates: List[str] = []
    if parsed:
        num, letter = parsed
        if letter:
            candidates.append(_canonical_diag_label(num, letter))  # e.g., 'Diag 5A'
        candidates.append(_canonical_diag_label(num, None))  # e.g., 'Diag 5'
    # also include raw text as a candidate
    candidates.append(str(diag_text).strip())
    try:
        logger.info(f"Diag candidates from notes '{diag_text}': {candidates}")
    except Exception:
        pass

    # Build normalized key map for magnets
    norm_to_key: Dict[str, str] = {}
    for k in magnets.keys():
        norm_to_key[_normalize_diag_key(k)] = k

    for cand in candidates:
        k_norm = _normalize_diag_key(cand)
        if k_norm in norm_to_key:
            k_real = norm_to_key[k_norm]
            try:
                logger.info(f"Matched diag candidate '{cand}' -> '{k_real}'")
            except Exception:
                pass
            return k_real, magnets[k_real]

    # As a last resort, try matching just the number when present
    if parsed:
        num, _ = parsed
        num_only = f"diag{num}"
        for nk, k_real in norm_to_key.items():
            if nk.startswith(num_only):  # matches diag5, diag5a, diag5b
                try:
                    logger.info(f"Fallback number-only match '{num_only}' -> '{k_real}'")
                except Exception:
                    pass
                return k_real, magnets[k_real]
    return None


@lru_cache(maxsize=256)
def _lookup_offsets_for_mold(mold_prefix: str) -> Optional[Tuple[str, Tuple[float, float, float, float]]]:
    """Given a mold prefix (e.g., 'H4a'), return (diag_id, (ox, oy, ix, iy)).

    Uses the 'mold directory' sheet col A (mold id) and col F (Notes) for the offset diag id,
    then looks it up in the Magnets sheet table.
    """
    try:
        if not mold_prefix:
            return None
        try:
            logger.info(f"Looking up offsets for mold prefix: {mold_prefix}")
        except Exception:
            pass
        xlsx = _device_master_list_path()
        # Resolve mold directory sheet name robustly
        mold_sheet = _resolve_sheet_name(
            xlsx_path=xlsx,
            primary_targets=["mold directory"],
            fallback_tokens=["mold", "directory"],
        )
        if mold_sheet is None:
            try:
                logger.warning("Failed to resolve mold directory sheet")
            except Exception:
                pass
            return None
        # Read only needed columns to be efficient
        try:
            md = pd.read_excel(
                xlsx,
                sheet_name=mold_sheet,
                usecols="A:F",
                engine=None,
            )
            try:
                logger.info(f"Loaded mold directory sheet '{mold_sheet}' with shape={md.shape}")
            except Exception:
                pass
        except Exception as e:
            try:
                logger.warning(f"Failed reading mold directory sheet '{mold_sheet}': {e}")
            except Exception:
                pass
            return None
        # Normalize headers if present; handle cases with/without header names gracefully
        col_a = md.columns[0]
        col_f = md.columns[5] if md.shape[1] >= 6 else md.columns[-1]
        # Find row(s) where mold id (col A) matches exactly the prefix
        try:
            mask = md[col_a].astype(str).str.strip() == str(mold_prefix).strip()
        except Exception:
            # Fallback to raw equality if astype/str fails
            mask = md[col_a] == mold_prefix
        rows = md.loc[mask]
        if rows.empty:
            try:
                logger.info(f"No row matched mold id '{mold_prefix}' in column A")
            except Exception:
                pass
            return None
        # Take the first match
        diag_id_raw = str(rows.iloc[0][col_f]).strip()
        if not diag_id_raw or diag_id_raw.lower() == "nan":
            try:
                logger.info("Notes (col F) empty for matched mold id")
            except Exception:
                pass
            return None
        try:
            logger.info(f"Notes value for mold '{mold_prefix}': '{diag_id_raw}'")
        except Exception:
            pass
        found = _find_offsets_by_diag_identifier(diag_id_raw)
        if found is None:
            try:
                logger.info("No offsets matched for parsed notes value")
            except Exception:
                pass
            return None
        matched_key, offsets = found
        try:
            logger.info(f"Resolved offsets: {matched_key} -> {offsets}")
        except Exception:
            pass
        return matched_key, offsets
    except Exception as e:
        try:
            logger.warning(f"Offset lookup error: {e}")
        except Exception:
            pass
        return None


def _select_load_cells_dialog(root: Path) -> List[str]:
    """Show a simple Qt dialog listing subfolders under root, allowing multi-select.
    Returns a list of selected folder names. Empty list if cancelled or Qt unavailable.
    """
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
    max_bins: int,
    rebuild_toggles,
) -> None:
    """Attach "Add Load Cell to Plot" toolbar action to select one/many folders
    from Load_Cell_Runs and add their lines to the plot with toggles and remove 'x'.
    """
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
                    runs = _load_and_process_cell(root / name, params)
                    if runs:
                        per_cell_runs[name] = runs

                # Add artists for each newly present cell
                cols_local = ["x", "y", "z", "mag"]
                axes_map_local = {"x": (0, 0), "y": (0, 1), "z": (1, 0), "mag": (1, 1)}
                for col in cols_local:
                    meas_col, truth_col = _pair_for_axis(col)
                    r, c = axes_map_local[col]
                    ax = axes[r][c]
                    for cell_name in selected:
                        runs = per_cell_runs.get(cell_name, [])
                        if not runs:
                            continue
                        color = cell_to_color.get(cell_name, (0.2, 0.4, 0.8))
                        for variant, linestyle, alpha in (
                            ("sum", "-", 0.18),
                            ("inner", "--", 0.14),
                            ("outer", ":", 0.12),
                        ):
                            v_col = _variant_column(meas_col, variant)
                            if v_col not in runs[0].columns:
                                continue
                            res = _compute_binned_mean_std(runs, v_col, truth_col, n_bins=max_bins)
                            if res is None:
                                continue
                            centers, means, stds = res
                            line, = ax.plot(centers, means, color=color, linewidth=2.0, linestyle=linestyle, label=f"{cell_name} {variant}")
                            band = ax.fill_between(centers, means - stds, means + stds, color=color, alpha=alpha)
                            key = (cell_name, variant)
                            grouped_artists.setdefault(key, []).extend([line, band])
                            # Only 'sum' visible by default for new cells
                            default_visible = (variant == "sum")
                            grouped_active.setdefault(key, default_visible)
                            try:
                                line.set_visible(default_visible)
                                band.set_visible(default_visible)
                            except Exception:
                                pass
                            group_to_labels.setdefault(cell_name, [])
                            if variant not in group_to_labels[cell_name]:
                                group_to_labels[cell_name].append(variant)

                # Rebuild toggles to include newly added cells and remove buttons
                rebuild_toggles()

                fig.canvas.draw_idle()
            except Exception:
                pass

        action.triggered.connect(_on_add_cells)
    except Exception:
        pass

def plot_load_cells_measured_vs_truth(
    per_cell_runs: Dict[str, List[pd.DataFrame]],
    params: Params,
    max_bins: int = 60,
    title_suffix: str = "",
    save_path: Optional[Path] = None,
    show: bool = True,
) -> None:
    sns.set_style("whitegrid")
    cols = ["x", "y", "z", "mag"]
    fig, axes = plt.subplots(2, 2, figsize=(16, 9))
    axes_map = {"x": (0, 0), "y": (0, 1), "z": (1, 0), "mag": (1, 1)}

    # Distinguish load cells by color, variants by linestyle. Start empty; assign on demand.
    cell_names = list(per_cell_runs.keys())
    palette = sns.color_palette("tab10", n_colors=10)
    cell_to_color: Dict[str, Tuple[float, float, float]] = {
        name: palette[i % len(palette)] for i, name in enumerate(cell_names)
    }

    # Collect artists for optional toggles
    # Grouped by load cell (and an 'overall' group) with per-variant toggles
    grouped_artists: Dict[Tuple[str, str], List] = {}
    grouped_active: Dict[Tuple[str, str], bool] = {}
    group_to_labels: Dict[str, List[str]] = {}

    # Precompute OVERALL lines from disk using cell-wise averaging
    overall_data = _compute_overall_cellwise_from_disk(params, n_bins=max_bins)

    for col in cols:
        meas_col, truth_col = _pair_for_axis(col)
        r, c = axes_map[col]
        ax = axes[r][c]
        meas_label, truth_label = _pair_for_axis_labels(col)
        # Match labeling style from plot_measured_vs_truth.py
        ax.set_xlabel(meas_label, fontsize=18)
        ax.set_ylabel(truth_label, fontsize=18)
        ax.tick_params(axis='both', which='major', labelsize=12)

        # Per-load-cell mean±SD bands (only for currently selected cells)
        for cell_name, runs in per_cell_runs.items():
            color = cell_to_color.get(cell_name, (0.2, 0.4, 0.8))
            for variant, linestyle, alpha in (
                ("sum", "-", 0.18),
                ("inner", "--", 0.14),
                ("outer", ":", 0.12),
            ):
                v_col = _variant_column(meas_col, variant)
                if v_col not in runs[0].columns:
                    # Skip this variant if not present in data
                    continue
                res = _compute_binned_mean_std(runs, v_col, truth_col, n_bins=max_bins)
                if res is None:
                    continue
                centers, means, stds = res
                line, = ax.plot(centers, means, color=color, linewidth=2.0, linestyle=linestyle, label=f"{cell_name} {variant}")
                band = ax.fill_between(centers, means - stds, means + stds, color=color, alpha=alpha)
                key = (cell_name, variant)
                grouped_artists.setdefault(key, []).extend([line, band])
                grouped_active.setdefault(key, True)
                if cell_name not in group_to_labels:
                    group_to_labels[cell_name] = []
                if variant not in group_to_labels[cell_name]:
                    group_to_labels[cell_name].append(variant)

        # Overall (across all cells on disk) mean±SD bands via cell-wise averaging
        for variant, linestyle, alpha in (("sum", "-", 0.20), ("inner", "--", 0.16), ("outer", ":", 0.14)):
            axis_data = overall_data.get(col, {})
            if variant not in axis_data:
                continue
            centers, means, stds = axis_data[variant]
            line, = ax.plot(centers, means, color="#222222", linewidth=2.3, linestyle=linestyle, label=f"overall {variant}")
            band = ax.fill_between(centers, means - stds, means + stds, color="#222222", alpha=alpha)
            key = ("overall", variant)
            grouped_artists.setdefault(key, []).extend([line, band])
            # Only 'sum' visible by default
            default_visible = (variant == "sum")
            grouped_active.setdefault(key, default_visible)
            try:
                line.set_visible(default_visible)
                band.set_visible(default_visible)
            except Exception:
                pass
            if "overall" not in group_to_labels:
                group_to_labels["overall"] = []
            if variant not in group_to_labels["overall"]:
                group_to_labels["overall"].append(variant)

        # Remove legend to avoid clutter; toggles on the right will serve as legend and controls
        # ax.legend(loc="best", fontsize=8)

    # Add padding between plots and the right-side toggles by shrinking plot area a bit
    fig.subplots_adjust(right=0.82, wspace=0.25, hspace=0.28)

    # Grouped toggles panels with dynamic rebuild and removable groups
    toggle_axes: List = []
    live_widgets: List = []  # Track CheckButtons and Buttons to disconnect events before removal
    live_cursors: List = []  # Track mplcursors.Cursor instances to enable/disable hover labels
    hover_enabled: List[bool] = [False]  # mutable flag so closures can see updates

    def rebuild_toggles() -> None:
        # 1) Disconnect events from any existing widgets to avoid callbacks after axes removal
        for w in list(live_widgets):
            try:
                if hasattr(w, "disconnect_events"):
                    w.disconnect_events()
            except Exception:
                pass
        live_widgets.clear()

        # 2) Remove previous toggle axes
        for ax_t in list(toggle_axes):
            try:
                ax_t.remove()
            except Exception:
                pass
        toggle_axes.clear()

        if not group_to_labels:
            return
        # ensure consistent variant order within each group
        variant_order = ["sum", "inner", "outer"]
        for g, lbls in list(group_to_labels.items()):
            ordered = [v for v in variant_order if v in lbls]
            ordered.extend([v for v in lbls if v not in ordered])
            group_to_labels[g] = ordered

        groups_in_order = [g for g in group_to_labels.keys() if g != "overall"]
        if "overall" in group_to_labels:
            groups_in_order.append("overall")

        total_items = sum(len(group_to_labels[g]) for g in groups_in_order)
        num_groups = len(groups_in_order)
        x0 = 0.86
        width = 0.12
        y_top = 0.92
        y_bottom = 0.12
        available = max(0.0, y_top - y_bottom)
        title_h = 0.02
        gap_h = 0.02
        denom = max(1, total_items)
        item_h = (available - num_groups * (title_h + gap_h)) / denom if available > 0 else 0.05
        item_h = min(0.07, max(0.025, item_h))

        chks: List[CheckButtons] = []
        style_map = {"sum": "-", "inner": "--", "outer": ":"}
        y_cursor = y_top

        for group_name in groups_in_order:
            labels = group_to_labels.get(group_name, [])
            n_items = len(labels)
            if n_items == 0:
                continue
            # Title
            try:
                title_offset = 0.008
                title_ax = fig.add_axes([x0, y_cursor - title_offset, width - 0.03, 0.02])
                title_ax.axis('off')
                title_ax.text(0, 0, group_name, fontsize=10, fontweight="bold", ha="left", va="bottom")
                toggle_axes.append(title_ax)
                # Remove 'x' button for non-overall groups
                if group_name != "overall":
                    rm_ax = fig.add_axes([x0 + width - 0.028, y_cursor - title_offset, 0.024, 0.02])
                    rm_btn = Button(rm_ax, "x")

                    def make_on_remove(gn: str):
                        def _on_remove(_event) -> None:
                            # Remove plot artists for this group on main axes
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
                            # Remove from dicts
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

            # Toggle box
            height = n_items * item_h
            y0 = y_cursor - title_h - height
            togg_ax = fig.add_axes([x0, y0, width, height])
            actives = [grouped_active.get((group_name, v), True) for v in labels]
            chk = CheckButtons(togg_ax, labels=labels, actives=actives)
            chks.append(chk)
            toggle_axes.append(togg_ax)
            live_widgets.append(chk)

            # style text and sample lines
            try:
                for txt in chk.labels:
                    txt.set_fontsize(9)
            except Exception:
                pass

            try:
                for txt, v in zip(chk.labels, labels):
                    trans = txt.get_transform()
                    x_txt, y_txt = txt.get_position()
                    x1 = min(0.80, x_txt + 0.20)
                    x2 = min(0.98, x1 + 0.16)
                    color = "#222222" if group_name == "overall" else cell_to_color.get(group_name, (0.2, 0.4, 0.8))
                    linestyle = style_map.get(v, "-")
                    line = plt.Line2D([x1, x2], [y_txt, y_txt], color=color, linestyle=linestyle, linewidth=2.6, solid_capstyle="round")
                    line.set_transform(trans)
                    line.set_zorder(2000)
                    line.set_clip_on(False)
                    line.set_visible(grouped_active.get((group_name, v), True))
                    togg_ax.add_line(line)
                    grouped_artists.setdefault((group_name, v), []).append(line)
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
                    for a in grouped_artists.get((gn, label), []):
                        try:
                            a.set_visible(visible)
                        except Exception:
                            pass
                    try:
                        fig.canvas.draw_idle()
                    except Exception:
                        pass
                return on_toggle

            chk.on_clicked(_make_cb(group_name, labels, chk))
            y_cursor = y0 - gap_h

        # Global OFF
        try:
            all_off_ax = fig.add_axes([x0, max(y_bottom - 0.02, 0.02), width, 0.06])
            all_off_btn = Button(all_off_ax, "All OFF")
            toggle_axes.append(all_off_ax)
            live_widgets.append(all_off_btn)

            def on_all_off(_event) -> None:
                for chk in chks:
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
                try:
                    fig.canvas.draw_idle()
                except Exception:
                    pass

            all_off_btn.on_clicked(on_all_off)
        except Exception:
            pass

    # Initial build of toggles
    rebuild_toggles()

    # Function to recompute overall from disk and refresh only overall artists
    def recompute_overall_and_refresh() -> None:
        try:
            new_overall = _compute_overall_cellwise_from_disk(params, n_bins=max_bins)
            main_axes = [ax for ax in getattr(axes, "ravel", lambda: [])()] if hasattr(axes, "ravel") else list(axes)
            # 1) Remove all previous overall artists from the main axes ONCE, keep toggle sample lines
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

            # 2) Add refreshed overall lines to ALL axes
            for col in cols:
                r, c = axes_map[col]
                ax = axes[r][c]
                axis_data = new_overall.get(col, {})
                for variant, linestyle, alpha in (("sum", "-", 0.20), ("inner", "--", 0.16), ("outer", ":", 0.14)):
                    if variant not in axis_data:
                        continue
                    centers, means, stds = axis_data[variant]
                    line, = ax.plot(centers, means, color="#222222", linewidth=2.3, linestyle=linestyle, label=f"overall {variant}")
                    band = ax.fill_between(centers, means - stds, means + stds, color="#222222", alpha=alpha)
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

    # Attach toolbar actions (Qt backends only)
    _attach_add_runs_to_project_button(
        fig=fig,
        axes=axes,
        per_cell_runs=per_cell_runs,
        params=params,
        grouped_artists=grouped_artists,
        grouped_active=grouped_active,
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
        max_bins=max_bins,
        rebuild_toggles=rebuild_toggles,
    )

    # Toolbar toggle for hover labels (load cell names). Applies to visible cell lines only.
    try:
        manager = getattr(fig.canvas, "manager", None)
        toolbar = getattr(manager, "toolbar", None)
        if toolbar is not None and mplcursors is not None:
            try:
                toolbar.addSeparator()
            except Exception:
                pass
            hover_action = toolbar.addAction("Hover Labels (OFF)")
            try:
                hover_action.setCheckable(True)
                hover_action.setChecked(False)
            except Exception:
                pass

            # Map line artist -> group name for label lookup
            line_to_group: Dict[Line2D, str] = {}

            def _clear_cursors():
                # Disconnect and remove existing cursors
                for cur in list(live_cursors):
                    try:
                        cur.remove()
                    except Exception:
                        pass
                live_cursors.clear()

            def _get_visible_lines() -> List[Line2D]:
                # Visible per-cell lines (exclude overall, bands)
                line_to_group.clear()
                lines: List[Line2D] = []
                main_axes = [ax for ax in getattr(axes, "ravel", lambda: [])()] if hasattr(axes, "ravel") else list(axes)
                for (group_name, variant), arts in grouped_artists.items():
                    if group_name == "overall":
                        continue
                    if variant not in ("sum", "inner", "outer"):
                        continue
                    for a in arts:
                        if isinstance(a, Line2D) and getattr(a, "axes", None) in main_axes and a.get_visible():
                            lines.append(a)
                            line_to_group[a] = group_name
                return lines

            def _apply_cursors():
                if not hover_enabled[0]:
                    _clear_cursors()
                    return
                artists = _get_visible_lines()
                if not artists:
                    _clear_cursors()
                    return
                try:
                    cur = mplcursors.cursor(artists, hover=True)  # multiple=False by default → one label at a time
                    @cur.connect("add")
                    def on_add(sel):
                        try:
                            g = line_to_group.get(sel.artist, "")
                            label = g
                            if g and g != "overall":
                                try:
                                    mold_prefix = str(g).split(".")[0]
                                    try:
                                        logger.info(f"Hover over group '{g}', mold prefix '{mold_prefix}'")
                                    except Exception:
                                        pass
                                    res = _lookup_offsets_for_mold(mold_prefix)
                                    if res is not None:
                                        diag_id, (ox, oy, ix, iy) = res
                                        # Build concise multi-line label
                                        parts = [f"{g}", f"Mold {mold_prefix} — {diag_id}", f"outer(x,y)=({ox:.3g},{oy:.3g})", f"inner(x,y)=({ix:.3g},{iy:.3g})"]
                                        label = "\n".join(parts)
                                    else:
                                        try:
                                            logger.info("No offsets found for hover label; showing name only")
                                        except Exception:
                                            pass
                                except Exception:
                                    pass
                            sel.annotation.set_text(label)
                        except Exception:
                            pass
                    live_cursors.append(cur)
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

            # Keep hover labels in sync with visibility changes and rebuilds
            def _sync_after_draw(_evt=None):
                _apply_cursors()
            try:
                fig.canvas.mpl_connect('draw_event', _sync_after_draw)
            except Exception:
                pass

            # Hide labels when mouse leaves the plot area; recreate when entering
            def _on_motion(evt):
                try:
                    if not hover_enabled[0]:
                        return
                    if evt.inaxes is None:
                        # Outside axes: clear annotations
                        _clear_cursors()
                    else:
                        # Inside axes: ensure cursor exists
                        if not live_cursors:
                            _apply_cursors()
                except Exception:
                    pass
            try:
                fig.canvas.mpl_connect('motion_notify_event', _on_motion)
            except Exception:
                pass
    except Exception:
        pass

    suptitle = "Measured vs Truth by Load Cell"
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
    parser = argparse.ArgumentParser(description="Plot measured vs truth per load cell (mounted only), with mean±SD bands for sum/inner/outer.")
    parser.add_argument("--root", type=str, default=str(lc_paths.default_root), help="Root folder containing load cell folders (each with *.csv)")
    parser.add_argument("--auto-source", type=str, default=str(lc_paths.auto_source_root), help="Root folder to recursively scan for new CIR *.csv files to auto-populate Load_Cell_Runs")
    parser.add_argument("--load-cell", type=str, default=None, help="Single load cell folder name to plot (subdir under --root)")
    parser.add_argument("--max-files-per-cell", type=int, default=0, help="Optional cap on number of files per load cell (0 = no cap)")
    parser.add_argument("--bins", type=int, default=60, help="Number of bins for truth-axis binning")
    parser.add_argument("--save", action="store_true", help="Save figure to outputs/plots")
    parser.add_argument("--no-show", action="store_false", dest="show", default=True, help="Do not show figure interactively")
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
                # Example: Mounted1_CIRspiralR_H42.59_10.16.2025.csv
                # Take the second-to-last token (right before the final underscore)
                parts = csv_path.name.split("_")
                if len(parts) < 2:
                    continue
                load_cell_name = parts[-2]
                load_cell_name = load_cell_name.strip()
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
    params = Params()  # reuse defaults (alignment, rotation, filtering options)

    # Start with no selected load cells; user adds via toolbar
    per_cell_runs: Dict[str, List[pd.DataFrame]] = {}

    # per_cell_runs may be empty at startup by design; we still show overall-only

    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    save_path: Optional[Path] = None
    if args.save:
        save_dir = lc_paths.outputs_root
        save_path = save_dir / f"load_cells_measured_vs_truth_{timestamp}.png"

    title_suffix = args.load_cell if args.load_cell else ""
    plot_load_cells_measured_vs_truth(
        per_cell_runs=per_cell_runs,
        params=params,
        max_bins=int(args.bins),
        title_suffix=title_suffix,
        save_path=save_path,
        show=bool(args.show),
    )


if __name__ == "__main__":
    main()


