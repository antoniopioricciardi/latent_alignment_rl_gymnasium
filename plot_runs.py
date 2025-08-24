import os
import sys

# Pick a headless-friendly MuJoCo GL backend if none is set
if 'MUJOCO_GL' not in os.environ:
    if sys.platform.startswith('linux') and not os.getenv('DISPLAY'):
        os.environ['MUJOCO_GL'] = 'egl'
import csv
from typing import List, Dict
import matplotlib.pyplot as plt

RUNS_DIR = "runs"

# Keys we attempt to plot from train.csv if present
TRAIN_METRICS = [
    ("episode_reward", "Train Episode Reward"),
    ("batch_reward", "Batch Reward (mean)"),
    ("critic_loss", "Critic Loss"),
    ("actor_loss", "Actor Loss"),
    ("alpha_value", "Alpha"),
]

EVAL_METRICS = [
    ("episode_reward", "Eval Episode Reward"),
]

def read_csv(path: str) -> List[Dict[str, float]]:
    rows = []
    if not os.path.exists(path):
        return rows
    with open(path, "r") as f:
        reader = csv.DictReader(f)
        for r in reader:
            # convert to float where possible
            converted = {}
            for k,v in r.items():
                try:
                    converted[k] = float(v)
                except (ValueError, TypeError):
                    converted[k] = v
            rows.append(converted)
    return rows


def find_run_dirs(root: str):
    for exp in os.listdir(root):
        exp_path = os.path.join(root, exp)
        if not os.path.isdir(exp_path):
            continue
        for algo in os.listdir(exp_path):
            algo_path = os.path.join(exp_path, algo)
            if not os.path.isdir(algo_path):
                continue
            for seed in os.listdir(algo_path):
                seed_path = os.path.join(algo_path, seed)
                if not os.path.isdir(seed_path):
                    continue
                yield exp, algo, seed, seed_path


def plot_time_series(x_key: str, y_key: str, label: str, rows: List[Dict[str, float]], ax, smoothing: int = 1):
    if not rows or y_key not in rows[0]:
        return
    xs = [r.get(x_key, i) for i, r in enumerate(rows)]
    ys = [r[y_key] for r in rows]
    if smoothing > 1 and len(ys) >= smoothing:
        smoothed = []
        cumsum = [0.0]
        for v in ys:
            cumsum.append(cumsum[-1] + v)
        for i in range(len(ys)):
            start = max(0, i - smoothing)
            total = cumsum[i+1] - cumsum[start]
            smoothed.append(total / (i+1 - start))
        ys = smoothed
    ax.plot(xs, ys, label=label)


def main():
    run_dirs = list(find_run_dirs(RUNS_DIR))
    if not run_dirs:
        print("No runs found under 'runs/'")
        return

    for exp, algo, seed, path in run_dirs:
        train_csv = os.path.join(path, "train.csv")
        eval_csv = os.path.join(path, "eval.csv")
        train_rows = read_csv(train_csv)
        eval_rows = read_csv(eval_csv)

        if not train_rows and not eval_rows:
            continue

        print(f"Plotting {exp}/{algo}/{seed}")
        # Train metrics
        fig, axes = plt.subplots(len(TRAIN_METRICS), 1, figsize=(6, 3*len(TRAIN_METRICS)), sharex=True)
        if len(TRAIN_METRICS) == 1:
            axes = [axes]
        for ax, (k, title) in zip(axes, TRAIN_METRICS):
            plot_time_series("step", k, title, train_rows, ax, smoothing=10)
            ax.set_ylabel(title)
            ax.grid(alpha=0.3)
        axes[-1].set_xlabel("Environment Steps")
        fig.suptitle(f"Train Metrics: {exp} | {algo} | seed {seed}")
        fig.tight_layout(rect=[0,0,1,0.96])
        out_dir = os.path.join(path, "plots")
        os.makedirs(out_dir, exist_ok=True)
        fig.savefig(os.path.join(out_dir, "train_metrics.png"), dpi=150)
        plt.close(fig)

        # Eval metrics
        if eval_rows:
            fig, ax = plt.subplots(figsize=(6,4))
            plot_time_series("step", "episode_reward", "Eval Episode Reward", eval_rows, ax, smoothing=1)
            ax.set_ylabel("Episode Reward")
            ax.set_xlabel("Environment Steps")
            ax.set_title(f"Eval: {exp} | {algo} | seed {seed}")
            ax.grid(alpha=0.3)
            fig.tight_layout()
            fig.savefig(os.path.join(out_dir, "eval_episode_reward.png"), dpi=150)
            plt.close(fig)

if __name__ == "__main__":
    main()
