import json
import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).parent.parent
summary_path = BASE_DIR / "analysis_ieq" / "Qwen3-0.6B" / "scene_alpha_summary.json"
out_dir = "figures"
os.makedirs(out_dir, exist_ok=True)

with open(summary_path, "r") as f:
    data = json.load(f)

by_task = data["by_task_type"]
tasks = list(by_task.keys())

flat_acc = [by_task[t]["flat"]["accuracy"] for t in tasks]
hier_acc = [by_task[t]["hier"]["accuracy"] for t in tasks]
clus_acc = [by_task[t]["clustered"]["accuracy"] for t in tasks]

x = np.arange(len(tasks))
w = 0.25

plt.figure(figsize=(10, 4))
plt.bar(x - w, flat_acc, width=w, label="Flat")
plt.bar(x, hier_acc, width=w, label="Hierarchical")
plt.bar(x + w, clus_acc, width=w, label="Clustered")

plt.xticks(x, tasks, rotation=30, ha="right")
plt.ylim(0, 1.05)
plt.ylabel("Accuracy")
plt.title("Accuracy by Task Type (MVE)")
plt.legend()

plt.tight_layout()
plt.savefig(os.path.join(out_dir, "fig2_accuracy_by_task.png"), dpi=300)
plt.close()
