# Sensor Sensitivity

Concise, interactive scripts to visualize load-cell data.

- Paths are resolved relative to this folder; you can run from repo root or here.

## Scripts
- `scripts/overlay_mounted_vs_reseated.py`: Overlay time-series for mounted (blue) vs reseated (red), with optional truth overlays and mean±SD bands.
- `scripts/plot_measured_vs_truth.py`: Scatter measured vs truth per axis with per-set mean±SD bands over truth bins.
- `scripts/plot_measured_vs_truth_bz_x.py`: Scatter measured vs BZ on the x-axis for all measured axes.

## Quick start
```bash
pip install -r "Sensor Sensitivity/requirements.txt"
python "Sensor Sensitivity/scripts/overlay_mounted_vs_reseated.py"
```

Or from this folder:
```bash
pip install -r requirements.txt
python scripts/overlay_mounted_vs_reseated.py
```