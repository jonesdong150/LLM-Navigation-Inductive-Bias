import json
import os
from pathlib import Path
import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).parent.parent
summary_path = BASE_DIR / "analysis_ieq" / "Qwen3-0.6B" / "scene_alpha_summary.json"
out_dir = "figures"
os.makedirs(out_dir, exist_ok=True)

with open(summary_path, "r") as f:
    data = json.load(f)

by_task = data["by_task_type"]
tasks = list(by_task.keys())

delta = [
    by_task[t]["hier"]["accuracy"] - by_task[t]["flat"]["accuracy"]
    for t in tasks
]

plt.figure(figsize=(9, 4))
plt.bar(tasks, delta)
plt.axhline(0, linestyle="--")
plt.ylabel("delta Accuracy (Hier - Flat)")
plt.title("Structure Sensitivity Across Tasks")
plt.xticks(rotation=30, ha="right")

plt.tight_layout()
plt.savefig(os.path.join(out_dir, "fig3_delta_accuracy.png"), dpi=300)
plt.close()
