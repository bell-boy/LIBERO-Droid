"""Aggregate a complete paired run and produce overall and per-suite plots."""
import csv
import json
import sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")


def report(root, trials, filament=False):
    arms = [("classic", "Classic · MuJoCo", "#4066a5"), ("droid", "Droid · MuJoCo", "#c15c36")]
    if filament:
        arms.append(("droid_filament", "Droid · Filament", "#16806f"))
    output = root / "three-way" if filament else root
    output.mkdir(exist_ok=True)
    rows = [json.loads(p.read_text()) for v, _, _ in arms for p in sorted((root/v).glob("*/*/*/episode.json"))]
    expected = {(v,s,t,i) for v, _, _ in arms for s in SUITES for t in range(10) for i in range(trials)}
    keys = {(r["variant"],r["suite"],r["task_id"],r["trial"]) for r in rows}
    if keys != expected or len(rows) != len(expected):
        raise ValueError(f"Expected {len(expected)} matched results, found {len(keys)}")
    labels = ["Overall", "Spatial", "Object", "Goal", "Long"]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    summary = []
    for j, (variant, label, color) in enumerate(arms):
        for i, suite in enumerate(("all", *SUITES)):
            group = [r for r in rows if r["variant"] == variant and suite in ("all", r["suite"])]
            summary.append(dict(environment=variant, suite=suite, episodes=len(group),
                                SR=np.mean([r["success"] for r in group]), TP=np.mean([r["tp"] for r in group])))
        for ax, metric in zip(axes, ("SR","TP")):
            values = [r[metric] for r in summary if r["environment"] == variant]
            bars = ax.bar(np.arange(5)+(j-(len(arms)-1)/2)*(.8/len(arms)), values, .76/len(arms), color=color, label=label)
            ax.bar_label(bars, fmt="%.2f", fontsize=8, padding=3)
    for ax, title in zip(axes, ("Success rate","Mean task progress")):
        ax.set(title=title, xticks=range(5), xticklabels=labels, ylim=(0,1.1), yticks=np.linspace(0,1,6))
        ax.spines[["top","right"]].set_visible(False)
        ax.set_axisbelow(True)
        ax.grid(axis="y", alpha=.18)
    fig.legend(*axes[0].get_legend_handles_labels(), loc="lower center", ncols=len(arms), frameon=False)
    fig.suptitle(f"π0.5 DROID joint-position policy  ·  40 tasks  ·  {trials} paired starts per task")
    fig.tight_layout(rect=(0,.09,1,.95))
    fig.savefig(output/"comparison.png", dpi=200)
    with (output/"summary.csv").open("w") as f:
        writer = csv.DictWriter(f, fieldnames=summary[0].keys())
        writer.writeheader()
        writer.writerows(summary)


if __name__ == "__main__":
    report(Path(sys.argv[1]), int(sys.argv[2]), "--filament" in sys.argv[3:])
