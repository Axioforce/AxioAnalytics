# AxioAnalytics

Analysis scripts for load cell variability and visualization.

## Structure
- `scripts/`: analysis and visualization scripts
- `Load_Cell_Spiral_test/`: raw CSV data (optional to track)
- `outputs/`: generated metrics and plots

## Quick start

```bash
pip install -r requirements.txt
python scripts/analyze_load_cell_variability.py
```

## Notes
- Consider using Git LFS if you want to version large `.csv` files.
- Outputs in `outputs/plots/` are ignored by default.
