import json
import os
from pathlib import Path
import matplotlib.pyplot as plt

# paths
BASE_DIR = Path(__file__).parent.parent
summary_path = BASE_DIR / "analysis_ieq" / "Qwen3-0.6B" / "scene_alpha_summary.json"
out_dir = "figures"
os.makedirs(out_dir, exist_ok=True)

with open(summary_path, "r") as f:
    data = json.load(f)

labels = ["Flat", "Hierarchical", "Clustered"]
acc = [
    data["overall"]["flat"]["accuracy"],
    data["overall"]["hier"]["accuracy"],
    data["overall"]["clustered"]["accuracy"],
]

plt.figure(figsize=(5, 4))
plt.bar(labels, acc)
plt.ylim(0, 1.05)
plt.ylabel("Accuracy")
plt.title("Overall Accuracy by Representation")

plt.tight_layout()
plt.savefig(os.path.join(out_dir, "fig1_overall_accuracy.png"), dpi=300)
plt.close()
