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

    avg_help = []
    avg_harm = []
    avg_hh = []

    for t in range(max_turns):
        help_scores = []
        harm_scores = []
        hh_scores = []

        for r in results:
            if t < len(r["turns"]):
                s = r["turns"][t]["scores"]
                # helpfulness
                if "helpfulness" in s:
                    help_scores.append(s["helpfulness"])

                # harmlessness: prefer explicit key, otherwise infer from "harmfulness" (inverse on 0-5 scale)
                if "harmlessness" in s:
                    harm_scores.append(s["harmlessness"])
                elif "harmfulness" in s:
                    try:
                        harm_scores.append(5.0 - float(s["harmfulness"]))
                    except Exception:
                        pass

                # hh_score
                if "hh_score" in s:
                    hh_scores.append(s["hh_score"])

        def mean(arr):
            if arr:
                return float(np.mean(arr))
            return 0.0

        mh = mean(help_scores)
        mm = mean(harm_scores)
        mhh = mean(hh_scores)

        avg_help.append(mh)
        avg_harm.append(mm)
        avg_hh.append(mhh)

    return avg_help, avg_harm, avg_hh


# ──────────────────────────────────────────────
# PLOT
# ──────────────────────────────────────────────

def plot_single(results, label=None):
    avg_help, avg_harm, avg_hh = compute_avg_scores(results)
    x = list(range(len(avg_help)))

    plt.plot(x, avg_help, marker='o', label='Helpfulness', color='#1f77b4')

    plt.plot(x, avg_harm, marker='o', label='Harmlessness', color='#2ca02c')

    plt.plot(x, avg_hh, marker='o', label='HH', color='#ff7f0e')


def plot_multiple(files, labels=None):
    plt.figure(figsize=(8, 5))

    for i, f in enumerate(files):
        results = load_results(f)
        label = labels[i] if labels else f
        plot_single(results, label)

    plt.xlabel("Number of Revisions")
    plt.ylabel("Score")
    plt.title("Self-Refinement Performance")

    # if labels or len(files) > 1:
    #     plt.legend()
    plt.legend(frameon=True, fontsize=12)

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