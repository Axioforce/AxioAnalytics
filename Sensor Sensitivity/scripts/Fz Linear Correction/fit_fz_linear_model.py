import os
import argparse
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
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


@dataclass
class Paths:
    project_root: Path = Path(__file__).resolve().parent.parent
    runs_root: Path = project_root / "Load_Cell_Runs"
    out_root: Path = project_root / "outputs" / "metrics" / "fz_linear_model"


def _ensure_dir(p: Path) -> None:
    try:
        p.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass


# Logger
logger = logging.getLogger("sensor_sensitivity.fz_linear_model")
if not logger.handlers:
    try:
        logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    except Exception:
        pass


def _scan_cells(root: Path) -> Dict[str, str]:
    cells: Dict[str, str] = {}
    for p in sorted(root.glob("*/")):
        cells[p.name] = str(p / "*.csv")
    return cells


def _robust_fit(X_in: np.ndarray, y_in: np.ndarray, max_iter: int = 30, ridge_lambda: float = 0.0) -> Tuple[np.ndarray, float, float]:
    """Iteratively reweighted least squares with optional ridge on non-intercept coefficients.

    Returns (beta, rmse, r2)
    """
    X = np.asarray(X_in, dtype=float)
    y = np.asarray(y_in, dtype=float)
    m = min(X.shape[0], y.size)
    if m < 3:
        return np.full(X.shape[1], np.nan), np.nan, np.nan
    X = X[:m, :]
    y = y[:m]
    mask = np.isfinite(y) & np.all(np.isfinite(X), axis=1)
    X = X[mask]
    y = y[mask]
    if X.shape[0] < max(3, X.shape[1]):
        return np.full(X.shape[1], np.nan), np.nan, np.nan

    w = np.ones(X.shape[0], dtype=float)

    def _solve(Xm: np.ndarray, ym: np.ndarray, wm: np.ndarray) -> Tuple[np.ndarray, float, float]:
        # Weight rows
        sqw = np.sqrt(wm)[:, None]
        Xw = Xm * sqw
        yw = ym * sqw.ravel()
        if ridge_lambda > 0:
            p = Xm.shape[1]
            pen = np.eye(p, dtype=float)
            pen[0, 0] = 0.0  # no penalty on intercept
            X_aug = np.vstack([Xw, np.sqrt(ridge_lambda) * pen])
            y_aug = np.concatenate([yw, np.zeros(p, dtype=float)])
        else:
            X_aug = Xw
            y_aug = yw
        try:
            beta = np.linalg.lstsq(X_aug, y_aug, rcond=None)[0]
        except Exception:
            beta = np.full(Xm.shape[1], np.nan)
        yhat = Xm @ beta if np.all(np.isfinite(beta)) else np.full_like(ym, np.nan)
        resid = ym - yhat
        rmse = float(np.sqrt(np.nanmean(resid ** 2))) if np.any(np.isfinite(resid)) else np.nan
        try:
            ss_res = float(np.nansum(resid ** 2))
            ss_tot = float(np.nansum((ym - float(np.nanmean(ym))) ** 2))
            r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else np.nan
        except Exception:
            r2 = np.nan
        return beta, rmse, r2

    beta, rmse, r2 = _solve(X, y, w)
    if not np.all(np.isfinite(beta)):
        return beta, rmse, r2
    for _ in range(max_iter):
        yhat = X @ beta
        resid = y - yhat
        s = 1.4826 * np.median(np.abs(resid - np.median(resid))) if resid.size else 0.0
        if not np.isfinite(s) or s <= 1e-12:
            break
        k = 1.345 * s
        new_w = np.clip(k / np.maximum(k, np.abs(resid)), 0.05, 1.0)
        if np.allclose(new_w, w, atol=1e-3, rtol=1e-3):
            break
        w = new_w
        beta, rmse, r2 = _solve(X, y, w)
        if not np.all(np.isfinite(beta)):
            break
    return beta, rmse, r2


def _preprocess_runs(glob_pattern: str, params: Params) -> List[pd.DataFrame]:
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


def _fit_per_run(df: pd.DataFrame, truth_cap_min: Optional[float], truth_cap_max: Optional[float], model: str, ridge_lambda: float = 1e-3) -> Dict[str, float]:
    out: Dict[str, float] = {"model": model, "a": np.nan, "b": np.nan, "b1": np.nan, "b2": np.nan, "c": np.nan, "c1": np.nan, "c2": np.nan, "rmse": np.nan, "r2": np.nan, "n": 0}
    # Require truth columns
    if not all(col in df.columns for col in ("bx", "by", "bz")):
        return out
    # Measured Fz against truth B
    if "z" not in df.columns:
        return out
    bx = df["bx"].to_numpy(dtype=float)
    by = df["by"].to_numpy(dtype=float)
    bz = df["bz"].to_numpy(dtype=float)
    fz = df["z"].to_numpy(dtype=float)
    br = np.sqrt(bx ** 2 + by ** 2)

    # Optional truth gating
    mask = np.isfinite(bx) & np.isfinite(by) & np.isfinite(bz) & np.isfinite(fz)
    if truth_cap_min is not None:
        mask &= (bz >= float(truth_cap_min))
    if truth_cap_max is not None:
        mask &= (bz <= float(truth_cap_max))

    if not np.any(mask):
        return out
    bz_use = bz[mask]
    br_use = br[mask]
    fz_use = fz[mask]

    if model == "linear":
        X = np.column_stack([np.ones_like(bz_use), bz_use, br_use])
        beta, rmse, r2 = _robust_fit(X, fz_use, ridge_lambda=0.0)
        if np.all(np.isfinite(beta)):
            out.update({"a": float(beta[0]), "b": float(beta[1]), "c": float(beta[2]), "rmse": rmse, "r2": r2, "n": int(np.sum(mask))})
        return out

    # lateral_curv model with orthogonalized r and centering of Bz
    bz_mean = float(np.nanmean(bz_use)) if bz_use.size else 0.0
    bz_c = bz_use - bz_mean
    # regress r on Bz_c: r_c ~ alpha0 + alpha1 * Bz_c
    try:
        A = np.column_stack([np.ones_like(bz_c), bz_c])
        alpha = np.linalg.lstsq(A, br_use, rcond=None)[0]
        r_hat = A @ alpha
    except Exception:
        r_hat = np.zeros_like(br_use)
    r_perp = br_use - r_hat

    if model == "lateral_curv":
        X = np.column_stack([np.ones_like(bz_c), bz_c, r_perp, r_perp ** 2])
        beta, rmse, r2 = _robust_fit(X, fz_use, ridge_lambda=ridge_lambda)
        if np.all(np.isfinite(beta)):
            out.update({
                "a": float(beta[0]),
                "b": float(beta[1]),
                "c1": float(beta[2]),
                "c2": float(beta[3]),
                "rmse": rmse,
                "r2": r2,
                "n": int(np.sum(mask)),
            })
        return out

    # z_curv model: [1, Bz_c, Bz_c^2, r_perp]
    X = np.column_stack([np.ones_like(bz_c), bz_c, bz_c ** 2, r_perp])
    beta, rmse, r2 = _robust_fit(X, fz_use, ridge_lambda=ridge_lambda)
    if np.all(np.isfinite(beta)):
        out.update({
            "a": float(beta[0]),
            "b1": float(beta[1]),
            "b2": float(beta[2]),
            "c1": float(beta[3]),
            "rmse": rmse,
            "r2": r2,
            "n": int(np.sum(mask)),
        })
    return out


def _aggregate_cell(rows: List[Dict[str, object]]) -> Dict[str, float]:
    if not rows:
        return {"a": np.nan, "b": np.nan, "c": np.nan, "c1": np.nan, "c2": np.nan, "w_mean_rmse": np.nan, "runs": 0}
    # Determine model variant present
    has_zcurv = any(np.isfinite(r.get("b2", np.nan)) for r in rows)
    if has_zcurv:
        coeffs = np.array([[r.get("a", np.nan), r.get("b1", np.nan), r.get("b2", np.nan), r.get("c1", np.nan), r.get("rmse", np.nan)] for r in rows], dtype=float)
        a_arr, b1_arr, b2_arr, c1_arr, rmse_arr = coeffs.T
    else:
        # lateral_curv or linear
        use_c1c2 = any(np.isfinite(r.get("c1", np.nan)) or np.isfinite(r.get("c2", np.nan)) for r in rows)
        if use_c1c2:
            coeffs = np.array([[r.get("a", np.nan), r.get("b", np.nan), r.get("c1", np.nan), r.get("c2", np.nan), r.get("rmse", np.nan)] for r in rows], dtype=float)
            a_arr, b_arr, c1_arr, c2_arr, rmse_arr = coeffs.T
        else:
            coeffs = np.array([[r.get("a", np.nan), r.get("b", np.nan), r.get("c", np.nan), r.get("rmse", np.nan)] for r in rows], dtype=float)
            a_arr, b_arr, c_arr, rmse_arr = coeffs.T
    # weights: 1/(rmse^2 + eps)
    eps = 1e-6
    w = 1.0 / np.maximum(rmse_arr ** 2 + eps, eps)
    w[~np.isfinite(w)] = 0.0
    def wmean(v: np.ndarray) -> float:
        num = np.nansum(w * v)
        den = np.nansum(w)
        return float(num / den) if den > 0 else np.nan
    if has_zcurv:
        agg = {
            "a": wmean(a_arr),
            "b1": wmean(b1_arr),
            "b2": wmean(b2_arr),
            "c1": wmean(c1_arr),
            "w_mean_rmse": wmean(rmse_arr),
            "runs": int(len(rows)),
            "model": "z_curv",
        }
    elif 'c1_arr' in locals() and 'c2_arr' in locals():
        agg = {
            "a": wmean(a_arr),
            "b": wmean(b_arr),
            "c1": wmean(c1_arr),
            "c2": wmean(c2_arr),
            "w_mean_rmse": wmean(rmse_arr),
            "runs": int(len(rows)),
            "model": "lateral_curv",
        }
    else:
        agg = {
            "a": wmean(a_arr),
            "b": wmean(b_arr),
            "c": wmean(c_arr),
            "w_mean_rmse": wmean(rmse_arr),
            "runs": int(len(rows)),
            "model": "linear",
        }
    return agg


def run_fit(
    runs_root: Path,
    out_root: Path,
    alignment: str = "local_peak",
    rotation_deg_z: float = -45.0,
    truth_cap_min: Optional[float] = None,
    truth_cap_max: Optional[float] = None,
    model: str = "z_curv",
    ridge_lambda: float = 1e-3,
) -> Dict[str, Path]:
    paths = Paths(project_root=runs_root.parent, runs_root=runs_root, out_root=out_root)
    params = Params()
    params.alignment_method = alignment
    params.rotation_deg_z = float(rotation_deg_z)

    logger.info(f"Scanning cells under: {paths.runs_root}")
    cells = _scan_cells(paths.runs_root)
    if not cells:
        logger.warning("No cells found. Nothing to fit.")
        return {}
    logger.info(f"Found {len(cells)} cell(s)")
    run_rows: List[Dict[str, object]] = []
    for cell, gpat in cells.items():
        logger.info(f"Loading runs for cell '{cell}'")
        dfs = _preprocess_runs(gpat, params)
        logger.info(f"  Loaded {len(dfs)} run(s)")
        for idx, df in enumerate(dfs, start=1):
            if (idx == 1) or (idx % 5 == 0) or (idx == len(dfs)):
                logger.info(f"  Fitting run {idx}/{len(dfs)}")
            src = str(df["source_path"].iloc[0]) if "source_path" in df.columns and len(df) else ""
            res = _fit_per_run(df, truth_cap_min, truth_cap_max, model=model, ridge_lambda=ridge_lambda)
            run_rows.append({
                "cell": cell,
                "run_file": src,
                **res,
            })

    if not run_rows:
        return {}

    _ensure_dir(paths.out_root)
    runs_df = pd.DataFrame(run_rows)
    runs_csv = paths.out_root / "runs.csv"
    runs_df.to_csv(runs_csv, index=False)
    logger.info(f"Wrote per-run coefficients: {runs_csv}")

    # Aggregate per cell
    agg_rows: List[Dict[str, object]] = []
    for cell, sub in runs_df.groupby("cell"):
        rows = sub.to_dict(orient="records")
        agg = _aggregate_cell(rows)
        agg_rows.append({"cell": cell, **agg})
    agg_df = pd.DataFrame(agg_rows)
    agg_csv = paths.out_root / "per_cell_overall.csv"
    agg_df.to_csv(agg_csv, index=False)
    logger.info(f"Wrote per-cell aggregated coefficients: {agg_csv}")

    # Fleet-wide weighted average across all runs
    fleet_agg = _aggregate_cell(runs_df.to_dict(orient="records"))
    fleet_df = pd.DataFrame([fleet_agg])
    fleet_csv = paths.out_root / "fleet_overall.csv"
    fleet_df.to_csv(fleet_csv, index=False)
    logger.info(f"Wrote fleet-wide coefficients: {fleet_csv}")

    return {"runs": runs_csv, "per_cell_overall": agg_csv, "fleet_overall": fleet_csv}


def main() -> None:
    p = Paths()
    ap = argparse.ArgumentParser(description="Fit Fz = a + b*bz + c*sqrt(bx^2+by^2) per run and aggregate per cell")
    ap.add_argument("--root", type=str, default=str(p.runs_root), help="Root folder containing per-cell subfolders of *.csv runs")
    ap.add_argument("--out", type=str, default=str(p.out_root), help="Output folder for metrics")
    ap.add_argument("--alignment", type=str, default="local_peak", choices=["frame", "event", "peak", "local_peak"], help="Alignment method")
    ap.add_argument("--rot-z", type=float, default=-45.0, help="Rotation about Z after alignment (deg)")
    ap.add_argument("--truth-min", type=float, default=None, help="Optional min bz (truth) to include in fit")
    ap.add_argument("--truth-max", type=float, default=5000.0, help="Optional max bz (truth) to include in fit")
    ap.add_argument("--model", type=str, default="z_curv", choices=["linear", "lateral_curv", "z_curv"], help="Model variant")
    ap.add_argument("--ridge", type=float, default=1e-3, help="Ridge penalty (non-intercept)")
    args = ap.parse_args()

    logger.info("Starting Fz linear model fit...")
    logger.info(f"Args: root={args.root}, out={args.out}, alignment={args.alignment}, rot_z={args.rot_z}, truth_min={args.truth_min}, truth_max={args.truth_max}, model={args.model}, ridge={args.ridge}")
    out = run_fit(
        runs_root=Path(args.root),
        out_root=Path(args.out),
        alignment=str(args.alignment),
        rotation_deg_z=float(args.rot_z),
        truth_cap_min=float(args.truth_min) if args.truth_min is not None else None,
        truth_cap_max=float(args.truth_max) if args.truth_max is not None else None,
        model=str(args.model),
        ridge_lambda=float(args.ridge),
    )
    if out:
        print("[fit_fz_linear_model] Wrote:")
        for k, v in out.items():
            print(f"  - {k}: {v}")
    else:
        print("[fit_fz_linear_model] No data found or nothing written.")


if __name__ == "__main__":
    main()


