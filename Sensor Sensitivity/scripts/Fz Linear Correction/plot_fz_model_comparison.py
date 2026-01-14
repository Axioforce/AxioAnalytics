import os
import argparse
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import logging

from pipeline import (
    Params,
    load_group,
    preprocess_signal,
    align_group_by_peak,
    align_group_by_local_peak,
    align_run,
    rotate_group_about_z,
)


# Logger
logger = logging.getLogger("sensor_sensitivity.plot_fz_model_comparison")
if not logger.handlers:
    try:
        logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    except Exception:
        pass


@dataclass
class Paths:
    project_root: Path = Path(__file__).resolve().parent.parent
    runs_root: Path = project_root / "Load_Cell_Runs"
    metrics_root: Path = project_root / "outputs" / "metrics" / "fz_linear_model"
    out_plots: Path = project_root / "outputs" / "plots" / "fz_model_comparison"


def _ensure_dir(p: Path) -> None:
    try:
        p.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass


def _bin_truth_mean_meas(x_truth: np.ndarray, y_meas: np.ndarray, n_bins: int = 60) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    x = np.asarray(x_truth, dtype=float)
    y = np.asarray(y_meas, dtype=float)
    m = min(x.size, y.size)
    if m == 0:
        return None
    x = x[:m]
    y = y[:m]
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if x.size == 0:
        return None
    xmin = float(np.nanmin(x))
    xmax = float(np.nanmax(x))
    if not np.isfinite(xmin) or not np.isfinite(xmax):
        return None
    if xmin == xmax:
        xmax = xmin + 1.0
    edges = np.linspace(xmin, xmax, int(n_bins) + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    means = np.full_like(centers, np.nan, dtype=float)
    for i in range(centers.size):
        sel = (x >= edges[i]) & (x < edges[i + 1])
        vals = y[sel]
        if vals.size:
            means[i] = float(np.nanmean(vals))
    return centers, means


def _load_one_run(paths: Paths, cell: str, run_index: Optional[int], run_file: Optional[str], params: Params) -> Optional[pd.DataFrame]:
    if run_file:
        gpat = run_file
        logger.info(f"Loading explicit run file: {gpat}")
        dfs = load_group(gpat, params.axis_mapping)
        if not dfs:
            logger.warning("No data loaded from run_file")
            return None
        raw = dfs[0]
        dfs_prep = [preprocess_signal(raw, params)]
    else:
        if not cell:
            logger.warning("--cell is required when --run-file is not provided")
            return None
        gpat = str(paths.runs_root / cell / "*.csv")
        logger.info(f"Scanning runs: {gpat}")
        dfs = load_group(gpat, params.axis_mapping)
        if not dfs:
            logger.warning("No runs found")
            return None
        idx = int(run_index) if run_index is not None else 0
        idx = max(0, min(idx, len(dfs) - 1))
        logger.info(f"Selected run index {idx+1}/{len(dfs)}")
        dfs_prep = [preprocess_signal(dfs[idx], params)]

    # Align and rotate like the analysis pipeline
    if params.alignment_method == "peak":
        dfs_aligned = align_group_by_peak(dfs_prep, params, signal="mag")
    elif params.alignment_method == "local_peak":
        dfs_aligned = align_group_by_local_peak(dfs_prep, params)
    else:
        dfs_aligned = [align_run(df, params) for df in dfs_prep]
    dfs_rot = rotate_group_about_z(dfs_aligned, params.rotation_deg_z)
    return dfs_rot[0] if dfs_rot else None


def _load_coefficients(paths: Paths, cell: Optional[str], source: str,
                       override_a: Optional[float], override_b: Optional[float], override_c: Optional[float]) -> Optional[Tuple[float, Dict[str, float], str]]:
    # Manual override wins
    if (override_a is not None) and (override_b is not None) and (override_c is not None):
        return float(override_a), {"b": float(override_b), "c": float(override_c)}, "override"
    try:
        if source == "per_cell" and cell:
            per_cell_csv = paths.metrics_root / "per_cell_overall.csv"
            df = pd.read_csv(per_cell_csv)
            row = df.loc[df["cell"] == cell]
            if not row.empty:
                a = float(row.iloc[0].get("a", np.nan))
                # accept linear (b,c) or lateral_curv (b,c1,c2)
                coeffs: Dict[str, float] = {}
                for key in ("b", "c", "c1", "c2"):
                    if key in row.columns:
                        coeffs[key] = float(row.iloc[0][key])
                if np.isfinite(a) and any(k in coeffs for k in ("c", "c1", "c2")):
                    return a, coeffs, "per_cell"
        # Fallback to fleet
        fleet_csv = paths.metrics_root / "fleet_overall.csv"
        df2 = pd.read_csv(fleet_csv)
        if not df2.empty:
            a = float(df2.iloc[0].get("a", np.nan))
            coeffs: Dict[str, float] = {}
            for key in ("b", "c", "c1", "c2"):
                if key in df2.columns:
                    coeffs[key] = float(df2.iloc[0][key])
            if np.isfinite(a) and any(k in coeffs for k in ("c", "c1", "c2")):
                return a, coeffs, "fleet"
    except Exception:
        pass
    return None


def plot_fz_model_comparison(
    cell: Optional[str],
    run_index: Optional[int],
    run_file: Optional[str],
    coeff_source: str,
    a: Optional[float], b: Optional[float], c: Optional[float],
    bins: int,
    flip_if_negative: bool,
    baseline_pct: float,
    save: bool,
    show: bool,
) -> Optional[Path]:
    paths = Paths()
    params = Params()  # reuse defaults

    df = _load_one_run(paths, cell or "", run_index, run_file, params)
    if df is None:
        logger.warning("No run loaded; aborting plot")
        return None
    if not all(col in df.columns for col in ("z", "bz")):
        logger.warning("Run lacks required columns 'z' and 'bz'")
        return None

    coeffs = _load_coefficients(paths, cell, coeff_source, a, b, c)
    if coeffs is None:
        logger.warning("Failed to load coefficients; aborting plot")
        return None
    a_hat, coeff_map, src = coeffs
    logger.info(f"Using coefficients from: {src} (a={a_hat:.6g}, keys={list(coeff_map.keys())})")

    # Prepare signals
    bx = df["bx"].to_numpy(dtype=float) if "bx" in df.columns else np.full(len(df), np.nan)
    by = df["by"].to_numpy(dtype=float) if "by" in df.columns else np.full(len(df), np.nan)
    bz = df["bz"].to_numpy(dtype=float)
    fz = df["z"].to_numpy(dtype=float)
    br = np.sqrt(bx ** 2 + by ** 2)

    # Support both linear (b,c) and lateral_curv (b,c1,c2 with orthogonalization approx)
    if ("c1" in coeff_map) or ("c2" in coeff_map):
        bz_mean = float(np.nanmean(bz)) if bz.size else 0.0
        bz_c = bz - bz_mean
        try:
            A = np.column_stack([np.ones_like(bz_c), bz_c])
            alpha = np.linalg.lstsq(A, br, rcond=None)[0]
            r_hat = A @ alpha
        except Exception:
            r_hat = np.zeros_like(br)
        r_perp = br - r_hat
        c1 = float(coeff_map.get("c1", 0.0))
        c2 = float(coeff_map.get("c2", 0.0))
        b_eff = float(coeff_map.get("b", 0.0))
        fz_hat = a_hat + b_eff * bz_c + c1 * r_perp + c2 * (r_perp ** 2)
    else:
        b_hat = float(coeff_map.get("b", 0.0))
        c_hat = float(coeff_map.get("c", 0.0))
        fz_hat = a_hat + b_hat * bz + c_hat * br

    # Bin by truth (bz) for clean curves
    raw_binned = _bin_truth_mean_meas(bz, fz, n_bins=bins)
    corr_binned = _bin_truth_mean_meas(bz, fz_hat, n_bins=bins)
    if raw_binned is None or corr_binned is None:
        logger.warning("Insufficient data after binning; aborting plot")
        return None
    x_raw, y_raw = raw_binned
    x_cor, y_cor = corr_binned

    # Optionally flip orientation to positive slope and re-anchor baseline
    flipped = False
    try:
        if flip_if_negative:
            valid = np.isfinite(x_cor) & np.isfinite(y_cor)
            if np.any(valid):
                slope = float(np.polyfit(x_cor[valid], y_cor[valid], 1)[0])
                if slope < 0:
                    flipped = True
                    fz_hat = -fz_hat
                    # re-bin after flipping
                    corr_binned = _bin_truth_mean_meas(bz, fz_hat, n_bins=bins)
                    if corr_binned is not None:
                        x_cor, y_cor = corr_binned
                    # re-anchor intercept using low-Bz region baseline
                    k = max(1, int(np.clip(baseline_pct, 0.01, 0.5) * len(x_cor)))
                    try:
                        baseline_raw = float(np.nanmedian(y_raw[:k]))
                        baseline_cor = float(np.nanmedian(y_cor[:k]))
                        delta = baseline_raw - baseline_cor
                        fz_hat = fz_hat + delta
                        corr_binned = _bin_truth_mean_meas(bz, fz_hat, n_bins=bins)
                        if corr_binned is not None:
                            x_cor, y_cor = corr_binned
                    except Exception:
                        pass
    except Exception:
        pass

    # Plot
    fig, ax = plt.subplots(1, 1, figsize=(9, 6))
    ax.plot(x_raw, y_raw, label="raw Fz vs Bz", color="#1f77b4", linewidth=2.0)
    label_cor = "corrected Fẑ vs Bz" + (" (flipped)" if flipped else "")
    ax.plot(x_cor, y_cor, label=label_cor, color="#d62728", linewidth=2.2)
    ax.set_xlabel(r"$B_z\ [\mu\mathrm{T}]$")
    ax.set_ylabel(r"$F_z\ [\mathrm{N}]$")
    title = "Fz linear model comparison"
    if cell:
        title += f" — {cell}"
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()

    out_path: Optional[Path] = None
    if save:
        _ensure_dir(paths.out_plots)
        # Construct a filename
        name_bits: List[str] = []
        if cell:
            name_bits.append(cell)
        if run_file:
            name_bits.append(Path(run_file).stem)
        elif run_index is not None:
            name_bits.append(f"run{int(run_index)}")
        else:
            name_bits.append("run")
        name_bits.append(src)
        fname = "fz_model_comparison_" + "_".join([b for b in name_bits if b]) + ".png"
        out_path = paths.out_plots / fname
        try:
            fig.savefig(str(out_path), dpi=150, bbox_inches="tight")
            logger.info(f"Saved plot: {out_path}")
        except Exception:
            logger.warning("Failed to save plot")
    if show:
        plt.show()
    else:
        plt.close(fig)
    return out_path


def main() -> None:
    p = Paths()
    ap = argparse.ArgumentParser(description="Plot raw vs corrected Fz for a single run against truth Bz")
    ap.add_argument("--cell", type=str, default=None, help="Cell folder name under Load_Cell_Runs (ignored if --run-file provided)")
    ap.add_argument("--run-index", type=int, default=0, help="Index of run within the cell (0-based)")
    ap.add_argument("--run-file", type=str, default=None, help="Explicit path to a run CSV (overrides --cell/--run-index)")
    ap.add_argument("--coeff-source", type=str, default="per_cell", choices=["per_cell", "fleet"], help="Coefficient source preference")
    ap.add_argument("--a", type=float, default=None, help="Override coefficient a")
    ap.add_argument("--b", type=float, default=None, help="Override coefficient b")
    ap.add_argument("--c", type=float, default=None, help="Override coefficient c")
    ap.add_argument("--bins", type=int, default=60, help="Number of Bz bins for smoothing curves")
    ap.add_argument("--flip-if-negative", action="store_true", help="Flip corrected curve to ensure positive slope vs Bz")
    ap.add_argument("--baseline-pct", type=float, default=0.15, help="Fraction of lowest-Bz bins to match baseline after flip (0-0.5)")
    ap.add_argument("--save", action="store_true", help="Save plot to outputs/plots")
    ap.add_argument("--no-show", action="store_false", dest="show", default=True, help="Do not show plot interactively")
    args = ap.parse_args()

    logger.info("Starting Fz model comparison plot...")
    plot_fz_model_comparison(
        cell=args.cell,
        run_index=args.run_index,
        run_file=args.run_file,
        coeff_source=args.coeff_source,
        a=args.a, b=args.b, c=args.c,
        bins=int(args.bins),
        flip_if_negative=bool(args.flip_if_negative),
        baseline_pct=float(args.baseline_pct),
        save=bool(args.save),
        show=bool(args.show),
    )


if __name__ == "__main__":
    main()


