# Benchmark figures

Run `python Visuals/plot_report.py` from the repository root. The script reads
the benchmark CSVs from `Data/` and writes five figures in PNG, SVG,
and PDF formats to `Visuals/images/`.

`Visuals/logos/` contains the artwork used in those figures. Python and NumPy
source logos are included as SVGs alongside their PNG renderings. The PyTorch,
Triton, and CUDA logos came from the project's original visuals.

Install plotting dependencies with
`python -m pip install matplotlib seaborn pandas Pillow`.
