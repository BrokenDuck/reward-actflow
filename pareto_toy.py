import re
import yaml
import matplotlib.pyplot as plt
from pathlib import Path

SCRATCH = Path("/cluster/scratch/kprotopapas/toy_sweep")
pattern = re.compile(r"^dps_([\d.]+)_ls_([\d.]+)$")

def load_metrics(path):
    eval_dir = path / "eval"
    iters = sorted(eval_dir.iterdir(), key=lambda x: int(x.name))
    for it in reversed(iters):
        f = it / "metrics.yaml"
        if f.exists():
            with open(f) as fh:
                return yaml.safe_load(fh)
    return None

BASELINE_WITH_FILTER = Path("/cluster/scratch/kprotopapas/baseline_with_filter")
BASELINE_NO_FILTER = Path("/cluster/scratch/kprotopapas/baseline_no_filter")

def to_point(metrics, label):
    return {
        "label": label,
        "valid": metrics["model_valid"],
        "coverage": metrics["coverage"],
        "vendi_linear": metrics["vendi_linear"],
        "vendi_rbf": metrics["vendi_rbf"],
    }

baseline_pt = to_point(load_metrics(BASELINE_WITH_FILTER), "baseline_with_filter")
baseline_nv_pt = to_point(load_metrics(BASELINE_NO_FILTER), "baseline_no_filter")

points = []
iter0_metrics = []
for d in sorted(SCRATCH.iterdir()):
    m = pattern.match(d.name)
    if not m:
        continue
    dps, ls = float(m.group(1)), float(m.group(2))
    eval_dir = d / "eval"
    if not eval_dir.exists():
        continue
    iters = sorted(eval_dir.iterdir(), key=lambda x: int(x.name))
    if not iters:
        continue
    metrics_file = iters[-1] / "metrics.yaml"
    if not metrics_file.exists():
        continue
    with open(metrics_file) as f:
        metrics = yaml.safe_load(f)
    points.append({
        "label": "sweep",
        "dps": dps, "ls": ls,
        "valid": metrics["model_valid"],
        "coverage": metrics["coverage"],
        "vendi_linear": metrics["vendi_linear"],
        "vendi_rbf": metrics["vendi_rbf"],
    })
    iter0_file = iters[0] / "metrics.yaml"
    if iter0_file.exists():
        with open(iter0_file) as f:
            iter0_metrics.append(yaml.safe_load(f))

def mean(vals):
    return sum(vals) / len(vals)

pretrained_pt = to_point({
    "model_valid": mean([m["model_valid"]    for m in iter0_metrics]),
    "coverage":    mean([m["coverage"]       for m in iter0_metrics]),
    "vendi_linear": mean([m["vendi_linear"]  for m in iter0_metrics]),
    "vendi_rbf":   mean([m["vendi_rbf"]      for m in iter0_metrics]),
}, "pre-trained")

def pareto_front(pts, x_key, y_key):
    sorted_pts = sorted(pts, key=lambda p: p[x_key], reverse=True)
    front, best_y = [], -float("inf")
    for p in sorted_pts:
        if p[y_key] > best_y:
            best_y = p[y_key]
            front.append(p)
    return list(reversed(front))

def normalize(pts, key, scale=1.0):
    vals = [p[key] for p in pts]
    lo, hi = min(vals), max(vals)
    if hi == lo:
        return {id(p): 0.0 for p in pts}
    return {id(p): (p[key] - lo) / (hi - lo) * scale for p in pts}

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

rows = [
    ("valid",    "Validity"),
    ("coverage", "Coverage"),
]

for col, (vendi_key, xlabel) in enumerate([
    ("vendi_linear", "Vendi (linear)"),
    ("vendi_rbf",    "Vendi (RBF)"),
]):
    sweep_pts = [p for p in points if p["valid"] >= 0.8]
    all_pts = sweep_pts + [baseline_pt, baseline_nv_pt, pretrained_pt]
    norm_vendi = normalize(all_pts, vendi_key, scale=100)

    for row, (y_key, ylabel) in enumerate(rows):
        ax = axes[row][col]
        norm_y = normalize(all_pts, y_key)
        front = pareto_front(sweep_pts, vendi_key, y_key)
        front_set = {(p["dps"], p["ls"]) for p in front}

        for p in sweep_pts:
            on_front = (p["dps"], p["ls"]) in front_set
            ax.scatter(norm_vendi[id(p)], norm_y[id(p)], color="steelblue",
                       alpha=1.0 if on_front else 0.15, zorder=3,
                       label="pareto front" if on_front else "_nolegend_")
            if on_front:
                ax.annotate(f"dps={p['dps']}\nls={p['ls']}",
                            (norm_vendi[id(p)], norm_y[id(p)]),
                            fontsize=7, textcoords="offset points", xytext=(4, 4))

        px = [norm_vendi[id(p)] for p in front]
        py = [norm_y[id(p)]     for p in front]
        ax.plot(px, py, color="steelblue", linewidth=1.5, label="_nolegend_")

        ax.scatter(norm_vendi[id(baseline_pt)], norm_y[id(baseline_pt)],
                   color="tomato", marker="*", s=150, zorder=5, label="baseline_with_filter")
        ax.scatter(norm_vendi[id(baseline_nv_pt)], norm_y[id(baseline_nv_pt)],
                   color="darkorange", marker="*", s=150, zorder=5, label="baseline (no verifier)")
        ax.scatter(norm_vendi[id(pretrained_pt)], norm_y[id(pretrained_pt)],
                   color="green", marker="*", s=150, zorder=5, label="pre-trained")

        if vendi_key == "vendi_linear":
            ax.set_xlim(95, 102)
        else:
            xmax = max(10, max(norm_vendi[id(p)] for p in front) * 1.05)
            ax.set_xlim(-0.1, xmax)
        ymin = 0.8 if y_key == "valid" else 0
        ax.set_ylim(ymin, 1.1)
        ax.set_xlabel(f"{xlabel} (normalised 0–100)")
        ax.set_ylabel(ylabel)
        ax.grid(True)

        handles, labels = ax.get_legend_handles_labels()
        seen, unique = set(), []
        for h, l in zip(handles, labels):
            if l not in seen:
                seen.add(l)
                unique.append((h, l))
        ax.legend(*zip(*unique), fontsize=8)

fig.tight_layout()
out = "pareto_toy.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
print(f"Saved to {out}")
