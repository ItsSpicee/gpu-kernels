GPU Kernels Exploratory Analysis

All benchmark functions return `(warmup_seconds, iteration_times_seconds)`
(PyTorch returns a dictionary with CPU/GPU entries in the same format).
CPU timings and synchronized GPU warm-ups use `time.perf_counter()` in seconds.
GPU samples use `triton.testing.do_bench_cudagraph()`, whose millisecond results
are converted to seconds immediately. Each GPU sample is an average over graph
replays, whereas each CPU sample measures one call. Warm-ups are measured separately
and excluded from sample statistics.

The summary and raw CSV writers convert seconds to milliseconds for all timing
columns. Regenerate existing CSVs with `python benchmark.py --skip-python`:
GPU sample timings exported before this fix may be inflated by 1,000; CPU and
warm-up timings were already in the correct units.

To repair the historical CSVs without rerunning the benchmarks:

```powershell
python correct_legacy_timings.py
```

This writes `benchmark_results_corrected.csv` and
`benchmark_raw_logs_corrected.csv`, preserving the originals. Only GPU execution
samples and their summary statistics are divided by 1,000. CPU timings and
warm-ups stay unchanged. Existing outputs are never overwritten. Use this script
only for old, mixed-unit results; new benchmark runs already export correct units.

To graph the corrected summary (requires Matplotlib):

```powershell
python -m pip install matplotlib
python plot_timings.py
```

The plot saves `benchmark_timings.png` and `benchmark_timings.svg`. It shows mean
execution time on a logarithmic axis, with one-standard-deviation whiskers and
sample counts. Warm-ups are excluded; single samples have no whiskers. For a new
benchmark run, plot its already-correct summary directly:

```powershell
python plot_timings.py --input benchmark_results.csv --output new_timings.png
```

Implements 7 different forms of Matrix Multiplication:

1. Pure Python CPU Implementation Using for loops and arrays
2. Numpy Built in MatMul
3. PyTorch Matmul
4. OpenAI's Triton naive MatMul (1 pid per output element)
5. Triton Blocked MatMul (1 pid per "block" of output elements)
6. Nvidia CUDA C++ Naive MatMul (1 thread per output element)
7. Nvidia CUDA C++ tiled MatMul (1 thread per output element, but groups threads into localized "blocks" that cooperatively load tiles of input matrices to increase gpu cache usage and decrease global memory traffic)

:):+

# Installation

### 1. Install Visual Studio 2022

Install Visual Studio 2022 Community with the following components:
- MSVC C++ compiler
- Windows SDK

### 2. Create the Conda Environment

```powershell
conda create -n gpukernels python=3.12
conda activate gpukernels
```

### 3. Install CUDA Toolkit 12.6

```powershell
conda install cuda-toolkit=12.6 -c nvidia
```

### 4. Install Python Dependencies

Install Triton for Windows:

```powershell
python -m pip install -U "triton-windows<3.3"
```

Install PyTorch with CUDA 12.6 support:

```powershell
python -m pip install torch --index-url https://download.pytorch.org/whl/cu126
```

Install NumPy and Ninja:

```powershell
python -m pip install numpy ninja
```

### 5. Configure the 64-bit Visual Studio Build Environment

Run the following command in PowerShell to initialize the Visual Studio 2022 developer environment for 64-bit compilation:

```powershell
& "C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\Tools\Launch-VsDevShell.ps1" -Arch amd64 -HostArch amd64
```

### 6. Run the Benchmarks

Navigate to the repository directory:

```powershell
cd path\to\gpu-kernels
```

Run the benchmarks, skipping the pure Python implementation:

```powershell
python benchmark.py --iterations 10 --skip-python
```

The benchmark produces two CSV files:

- `benchmark_results.csv` — summary statistics for each implementation.
- `benchmark_raw_logs.csv` — warm-up and individual iteration timings.

## CUDA Library Path Note

The CUDA library path in `cuda_bench_wrapper.py` is configured relative to the active Conda environment using `CONDA_PREFIX`:

```python
from pathlib import Path
import os

CUDA_LIB_DIR = str(
    Path(os.environ["CONDA_PREFIX"]) / "Library" / "lib"
)
```

The CUDA extension build passes this directory to the linker:

```python
extra_ldflags=[f"/LIBPATH:{CUDA_LIB_DIR}", "cudart.lib"],
```

This configuration worked for my setup because CUDA Toolkit 12.6 was installed inside the `gpukernels` Conda environment, where `cudart.lib` was located in `Library/lib`.

Users who do not use Conda may encounter a build error because `CONDA_PREFIX` may not be defined, or because their CUDA runtime library is installed in a different directory. The linker configuration assumes the library is located at the path constructed from the active Conda environment.
