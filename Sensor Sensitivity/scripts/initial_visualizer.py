import numpy as np
import pandas as pd
import sklearn
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_absolute_error
from pathlib import Path

# Resolve relative to the project folder (one level above this script)
project_root = Path(__file__).resolve().parent.parent
data_path = project_root / "Load_Cell_Spiral_test" / "H6b.90" / "10.08.2025" / "stationary_runs" / "stationary_spiral1_10.08.2025.csv"
data = pd.read_csv(data_path)

theta = np.deg2rad(-45)

Rz = np.array([[np.cos(theta), -np.sin(theta), 0],
               [np.sin(theta),  np.cos(theta), 0],
               [0, 0, 1]])

rawx= np.array([data['1-outer-x'],data['1-outer-y'],data['1-outer-z']]).T


x = rawx @ Rz.T


Y= np.array([data['bx'],data['by'],data['bz']]).T

fig, ax = plt.subplots(1, figsize=(6,6), dpi=30)
ax.set_xlabel("time")
ax.set_ylabel("data")
ax.scatter(data['time'], x[:, 0],  color='blue')
ax.scatter(data['time'], data['bx'],  color='cyan')
ax.scatter(data['time'], x[:, 1],  color='red')
ax.scatter(data['time'], data['by'],  color='pink')
#ax.scatter(data['time'], x[:, 2],  color='green')
#ax.scatter(data['time'], data['bz'],  color='olive')
fig.tight_layout() # this makes it look a little better
plt.show()