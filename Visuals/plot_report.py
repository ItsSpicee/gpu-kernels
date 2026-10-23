"""Create four publication figures from the benchmark CSV files.

Usage: python Visuals/plot_report.py
Dependencies: numpy, pandas, matplotlib, seaborn, Pillow.
Inputs must already contain correctly scaled milliseconds.
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from matplotlib.ticker import FuncFormatter
import numpy as np
import pandas as pd
from PIL import Image
import seaborn as sns

COLORS = {"Python": "#3776ab", "NumPy": "#4d77cf", "PyTorch": "#ee4c2c",
          "Triton": "#20a9b8", "CUDA": "#629b18"}
LOGOS = {"Python": "python-logo-official.png", "NumPy": "numpy-logo-official.png",
         "PyTorch": "pytorch-logo.png", "Triton": "triton-logo.png", "CUDA": "cuda-logo.jpg"}
INK, MUTED = "#172a3a", "#617385"
ASSETS = Path(__file__).resolve().parent
ROOT = ASSETS.parent
LOGOS_DIR = ASSETS / "logos"
IMAGES_DIR = ASSETS / "images"


def family(name):
    return name.split()[0]


def logo(ax, name, x, y, assets, size=32, coords="axes fraction"):
    image = np.asarray(Image.open(assets / LOGOS[family(name)]).convert("RGBA"))
    box = OffsetImage(image, zoom=size / max(image.shape[:2]))
    ax.add_artist(AnnotationBbox(box, (x, y), xycoords=coords, frameon=False,
                                annotation_clip=False, box_alignment=(0.5, 0.5)))


def finish(fig, target, note=None):
    if note:
        fig.text(.055, .038, note, fontsize=9, color=MUTED, va="bottom", linespacing=1.6)
    for suffix in ("png", "svg", "pdf"):
        fig.savefig(target.with_suffix("." + suffix), dpi=300, facecolor="white")
    plt.close(fig)
    print(f"Saved {target.name}: PNG, SVG, PDF")


def overview(summary, assets, output):
    rows = summary.sort_values("Mean (ms)")
    fig = plt.figure(figsize=(14, 8))
    fig.text(.055, .94, "Matrix multiplication across implementations", fontsize=25,
             weight="bold", color=INK, va="top")
    fig.text(.055, .885,
             "Mean execution time across recorded samples  ·  n=30 per implementation except Python (n=1)",
             fontsize=11, color=MUTED, va="top")
    ax = fig.add_axes([.32, .10, .53, .70])
    for i, (_, row) in enumerate(rows.iterrows()):
        name, mean = row["Implementation"], row["Mean (ms)"]
        color = COLORS[family(name)]
        if i % 2 == 0:
            ax.axhspan(i - .46, i + .46, color="#f3f7fa", zorder=0)
        ax.errorbar(mean, i, xerr=row["Std Dev (ms)"] if row["Iterations"] > 1 else None,
                    fmt="o", color=color, capsize=4, markersize=9, linewidth=2, zorder=3)
        logo(ax, name, -.46, i, assets, size=37, coords=ax.get_yaxis_transform())
        ax.text(-.37, i, name, transform=ax.get_yaxis_transform(), va="center",
                fontsize=12, weight="bold", color=INK)
        label = f"{mean / 1000:.2f} s" if mean >= 1000 else f"{mean:.4g} ms"
        ax.text(1.025, i, label, transform=ax.get_yaxis_transform(), va="center",
                fontsize=12, weight="bold", color=color)
    ax.set_ylim(len(rows) - .5, -.5)
    ax.set_xscale("log")
    ax.set_xlim(.06, 1e5)
    ax.set_xticks([.1, 1, 10, 100, 1000, 10000, 100000])
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:g}"))
    ax.set_yticks([])
    ax.set_xlabel("Execution time in milliseconds · logarithmic scale", labelpad=12)
    ax.grid(axis="x", alpha=.3)
    sns.despine(ax=ax, left=True)
    finish(fig, output / "01_overview")


def gpu_detail(summary, assets, output):
    fig = plt.figure(figsize=(14, 8))
    indexed = summary.set_index("Implementation")
    groups = [
        ("Triton vs PyTorch", ["Triton Blocked", "PyTorch GPU"], .15),
        ("CUDA implementations", ["CUDA Tiled", "CUDA Naive"], 1.25),
        ("Triton blocked vs naive", ["Triton Blocked", "Triton Naive"], 125),
    ]
    for k, (title, names, limit) in enumerate(groups):
        ax = fig.add_axes([.075 + k * .315, .22, .25, .58])
        for i, name in enumerate(names):
            row = indexed.loc[name]
            ax.bar(i, row["Mean (ms)"], width=.52, color=COLORS[family(name)], alpha=.9,
                   yerr=row["Std Dev (ms)"], capsize=5, error_kw={"ecolor": INK})
            ax.text(i, row["Mean (ms)"] + row["Std Dev (ms)"] + limit * .025,
                    f"{row['Mean (ms)']:.4g} ms", ha="center", fontsize=12,
                    weight="bold", color=INK)
            logo(ax, name, (i + .5) / len(names), -.19, assets, size=35)
        ax.set_xlim(-.5, len(names) - .5)
        ax.set_ylim(0, max(limit, max(indexed.loc[n, "Mean (ms)"] +
                                     indexed.loc[n, "Std Dev (ms)"] for n in names) * 1.2))
        if k == 2:
            # Keep a zero baseline while making the 0.1 ms blocked result visible
            # beside the 117 ms naive result.
            ax.set_yscale("symlog", linthresh=.1)
            ax.set_yticks([0, .1, 1, 10, 100], ["0", "0.1", "1", "10", "100"])
        ax.set_xticks(range(len(names)), names, fontsize=10)
        ax.set_title(title, loc="left", weight="bold", fontsize=13, pad=18)
        ax.set_ylabel("Mean execution time (ms), n=30")
        ax.grid(axis="y", alpha=.25)
        ax.set_axisbelow(True)
        sns.despine(ax=ax)
    finish(fig, output / "02_gpu_detail")


def distributions(summary, raw, assets, output):
    names = summary.loc[summary["Iterations"] > 1].sort_values("Median (ms)")["Implementation"].tolist()
    fig = plt.figure(figsize=(14, 10))
    rng = np.random.default_rng(42)
    for k, name in enumerate(names):
        col, row = k % 2, k // 2
        ax = fig.add_axes([.09 + .47 * col, .75 - .20 * row, .36, .13])
        values = raw.loc[(raw["Implementation"] == name) & (raw["Iteration"] > 0), "Time (ms)"]
        color = COLORS[family(name)]
        sns.boxplot(x=values, ax=ax, color=color, width=.42, showfliers=False,
                    boxprops={"alpha": .18}, linewidth=1.3)
        ax.scatter(values, rng.uniform(-.12, .12, len(values)), s=24, c=color,
                   alpha=.8, edgecolor="white", linewidth=.4, zorder=3)
        ax.set_ylim(.5, -.5)
        ax.set_yticks([])
        ax.set_xlabel("ms", fontsize=9, labelpad=0)
        ax.set_title(f"{name}  ·  median {np.median(values):.4g} ms", loc="left",
                     fontsize=11, weight="bold", color=INK, pad=10)
        logo(ax, name, -.09, .52, assets, size=31)
        ax.grid(axis="x", alpha=.2)
        sns.despine(ax=ax, left=True)
    finish(fig, output / "03_distributions")


def warmups(summary, assets, output):
    rows = summary.sort_values("Warmup (ms)")
    fig = plt.figure(figsize=(14, 8))
    fig.text(.055, .94, "Warm-up time across implementations", fontsize=25,
             weight="bold", color=INK, va="top")
    ax = fig.add_axes([.32, .12, .53, .73])
    for i, (_, row) in enumerate(rows.iterrows()):
        name, warmup = row["Implementation"], row["Warmup (ms)"]
        color = COLORS[family(name)]
        if i % 2 == 0:
            ax.axhspan(i - .46, i + .46, color="#f3f7fa", zorder=0)
        ax.plot(warmup, i, "o", color=color, markersize=9, zorder=3)
        logo(ax, name, -.46, i, assets, size=37, coords=ax.get_yaxis_transform())
        ax.text(-.37, i, name, transform=ax.get_yaxis_transform(), va="center",
                fontsize=12, weight="bold", color=INK)
        label = f"{warmup / 1000:.2f} s" if warmup >= 1000 else f"{warmup:.4g} ms"
        ax.text(1.025, i, label, transform=ax.get_yaxis_transform(), va="center",
                fontsize=12, weight="bold", color=color)
    ax.set_ylim(len(rows) - .5, -.5)
    ax.set_xscale("log")
    ax.set_xlim(.05, 1e5)
    ax.set_xticks([.1, 1, 10, 100, 1000, 10000, 100000])
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:g}"))
    ax.set_yticks([])
    ax.set_xlabel("Warm-up time in milliseconds · logarithmic scale", labelpad=12)
    ax.grid(axis="x", alpha=.3)
    sns.despine(ax=ax, left=True)
    finish(fig, output / "04_warmups")


def warmup_vs_mean(summary, assets, output):
    # The Python implementation has one measured sample, so its paired points
    # cannot show repeatability alongside implementations with 30 samples.
    rows = (summary.loc[summary["Iterations"] > 1]
            .sort_values("Mean (ms)").reset_index(drop=True))
    fig = plt.figure(figsize=(14, 8))
    fig.text(.055, .94, "Warm-up time vs mean execution time", fontsize=25,
             weight="bold", color=INK, va="top")
    ax = fig.add_axes([.32, .12, .53, .73])
    for i, row in rows.iterrows():
        name = row["Implementation"]
        color = COLORS[family(name)]
        warmup, mean = row["Warmup (ms)"], row["Mean (ms)"]
        if i % 2 == 0:
            ax.axhspan(i - .46, i + .46, color="#f3f7fa", zorder=0)
        ax.plot([warmup, mean], [i, i], color=color, alpha=.42, linewidth=2, zorder=1)
        ax.scatter(warmup, i, marker="o", s=75, color=color, edgecolor="white",
                   linewidth=.8, zorder=3)
        ax.scatter(mean, i, marker="D", s=68, color=color, edgecolor="white",
                   linewidth=.8, zorder=4)
        logo(ax, name, -.46, i, assets, size=37, coords=ax.get_yaxis_transform())
        ax.text(-.37, i, name, transform=ax.get_yaxis_transform(), va="center",
                fontsize=12, weight="bold", color=INK)
        ax.annotate(f"{warmup / 1000:.2f} s" if warmup >= 1000 else f"{warmup:.4g}",
                    (warmup, i), xytext=(0, 11), textcoords="offset points",
                    ha="center", fontsize=8, color=color)
        ax.annotate(f"{mean / 1000:.2f} s" if mean >= 1000 else f"{mean:.4g}",
                    (mean, i), xytext=(0, -14), textcoords="offset points",
                    ha="center", va="top", fontsize=8, color=color)
    ax.set_ylim(len(rows) - .5, -.5)
    ax.set_xscale("log")
    ax.set_xlim(.05, 1e5)
    ax.set_xticks([.1, 1, 10, 100, 1000, 10000, 100000])
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:g}"))
    ax.set_yticks([])
    ax.set_xlabel("Time (ms, logarithmic scale)  ·  circle: warm-up  ·  diamond: mean after warm-up",
                  labelpad=12)
    ax.grid(axis="x", alpha=.3)
    sns.despine(ax=ax, left=True)
    finish(fig, output / "05_warmup_vs_mean")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, default=ROOT / "Data" / "benchmark_results.csv")
    parser.add_argument("--raw", type=Path, default=ROOT / "Data" / "benchmark_raw_logs.csv")
    parser.add_argument("--assets", type=Path, default=LOGOS_DIR,
                        help="Directory containing the chart logos.")
    parser.add_argument("--output-dir", type=Path, default=IMAGES_DIR,
                        help="Directory for generated PNG, SVG, and PDF figures.")
    args = parser.parse_args()
    summary, raw = pd.read_csv(args.summary), pd.read_csv(args.raw)
    for _, row in summary.iterrows():
        values = raw.loc[(raw["Implementation"] == row["Implementation"]) & (raw["Iteration"] > 0), "Time (ms)"]
        if len(values) != row["Iterations"] or not np.isclose(values.mean(), row["Mean (ms)"]):
            raise ValueError(f"Summary/raw mismatch: {row['Implementation']}")
        if not np.isfinite(values).all() or (values <= 0).any():
            raise ValueError("Timings must be finite and positive.")
    sns.set_theme(style="white", font="DejaVu Sans", rc={"text.color": INK, "axes.labelcolor": MUTED,
                  "xtick.color": MUTED, "ytick.color": MUTED, "axes.edgecolor": "#cbd5df",
                  "svg.fonttype": "none", "pdf.fonttype": 42})
    args.output_dir.mkdir(parents=True, exist_ok=True)
    overview(summary, args.assets, args.output_dir)
    gpu_detail(summary, args.assets, args.output_dir)
    distributions(summary, raw, args.assets, args.output_dir)
    warmups(summary, args.assets, args.output_dir)
    warmup_vs_mean(summary, args.assets, args.output_dir)


if __name__ == "__main__":
    main()
