# Sensor Sensitivity

Analysis scripts for load cell variability and visualization.

This folder is self-contained and can be run from anywhere; paths are resolved relative to this folder.

## Structure
- `scripts/`: analysis and visualization scripts
- `Load_Cell_Spiral_test/`: raw CSV data
- `outputs/`: generated metrics and plots

## Quick start

From the repo root OR from this folder:

```bash
pip install -r "Sensor Sensitivity/requirements.txt"
python "Sensor Sensitivity/scripts/analyze_load_cell_variability.py"
```

Or, if your working directory is this folder:

```bash
pip install -r requirements.txt
python scripts/analyze_load_cell_variability.py
```