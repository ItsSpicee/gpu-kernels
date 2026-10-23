"""Plot a benchmark summary whose timing columns are already in milliseconds."""

import argparse
import csv
import math
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("benchmark_results_corrected.csv"))
    parser.add_argument("--output", type=Path, default=Path("benchmark_timings.png"))
    args = parser.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter

    with args.input.open(newline="", encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        parser.error("The summary contains no measurements.")
    for row in rows:
        for field in ("Mean (ms)", "Std Dev (ms)"):
            row[field] = float(row[field])
            if not math.isfinite(row[field]) or row[field] < 0:
                parser.error(f"Invalid {field} for {row['Implementation']}")
        if row["Mean (ms)"] == 0 or int(row["Iterations"]) < 1:
            parser.error("Means and sample counts must be positive.")
    rows.sort(key=lambda row: row["Mean (ms)"])

    fig, ax = plt.subplots(figsize=(11, max(4, len(rows) * 0.58 + 1.8)))
    for index, row in enumerate(rows):
        mean, std = row["Mean (ms)"], row["Std Dev (ms)"]
        # A single sample has no estimated standard deviation.
        error = std if int(row["Iterations"]) > 1 else None
        if error is not None and error >= mean:
            parser.error("Mean minus standard deviation must be positive for a log plot.")
        ax.errorbar(mean, index, xerr=error, fmt="o", color="#2563eb",
                    ecolor="#64748b", capsize=4, markersize=7)
        ax.text(1.02, index, f"{mean:.4g} ms", transform=ax.get_yaxis_transform(),
                va="center", fontsize=10)
    ax.set_yticks(range(len(rows)), [
        f"{row['Implementation']}  (n={row['Iterations']})" for row in rows
    ])
    ax.invert_yaxis()
    ax.set_xscale("log")
    ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))
    ax.set_xlabel("Execution time (ms, logarithmic scale) — lower is faster")
    ax.set_title("1024 × 1024 matrix multiplication", loc="left", fontweight="bold", pad=16)
    ax.grid(axis="x", which="major", alpha=0.3)
    ax.set_axisbelow(True)
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    fig.text(0.02, 0.025,
             "Points: mean • Whiskers: ±1 sample SD (not a confidence interval) • Warm-ups excluded\n"
             "CPU: wall-clock calls; GPU: CUDA graph timing aggregates. Precision settings may differ.",
             fontsize=9, color="#475569")
    fig.tight_layout(rect=(0, 0.1, 0.88, 1))
    fig.savefig(args.output, dpi=200, bbox_inches="tight")
    fig.savefig(args.output.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {args.output} and {args.output.with_suffix('.svg')}")


if __name__ == "__main__":
    main()
