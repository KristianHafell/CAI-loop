import json
import numpy as np
import matplotlib.pyplot as plt
import argparse


# ──────────────────────────────────────────────
# LOAD DATA
# ──────────────────────────────────────────────

def load_results(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ──────────────────────────────────────────────
# PROCESS DATA
# ──────────────────────────────────────────────

def compute_avg_scores(results):
    """
    Returns:
        avg_scores: list of mean HH score per turn
        std_scores: list of std deviation per turn
    """
    max_turns = max(len(r["turns"]) for r in results)

    avg_scores = []
    std_scores = []

    for t in range(max_turns):
        scores = []

        for r in results:
            if t < len(r["turns"]):
                hh = r["turns"][t]["scores"]["hh_score"]
                scores.append(hh)

        if scores:
            avg_scores.append(np.mean(scores))
            std_scores.append(np.std(scores))
        else:
            avg_scores.append(0)
            std_scores.append(0)

    return avg_scores, std_scores


# ──────────────────────────────────────────────
# PLOT
# ──────────────────────────────────────────────

def plot_single(results, label=None):
    avg, std = compute_avg_scores(results)
    x = list(range(len(avg)))

    plt.plot(x, avg, marker='o', label=label if label else "Run")
    plt.fill_between(x,
                     np.array(avg) - np.array(std),
                     np.array(avg) + np.array(std),
                     alpha=0.2)


def plot_multiple(files, labels=None):
    plt.figure(figsize=(8, 5))

    for i, f in enumerate(files):
        results = load_results(f)
        label = labels[i] if labels else f
        plot_single(results, label)

    plt.xlabel("Number of Revisions")
    plt.ylabel("HH Score")
    plt.title("Self-Refinement Performance")
    plt.grid(True)

    if labels or len(files) > 1:
        plt.legend()

    plt.tight_layout()
    plt.show()


# ──────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--files",
        nargs="+",
        required=True,
        help="List of results.json files"
    )
    parser.add_argument(
        "--labels",
        nargs="+",
        default=None,
        help="Optional labels for each file"
    )

    args = parser.parse_args()

    plot_multiple(args.files, args.labels)

# 
# ╔═══════════════════════════════════════════════════╗"
# ║   Run the plotter with e.g.:                      ║"
# ║                                                   ║"
# ║   python plot_results.py --files results.json     ║"
# ╚═══════════════════════════════════════════════════╝"
# 