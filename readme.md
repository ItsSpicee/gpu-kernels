GPU Kernels Exploratory Analysis


Implements 7 different forms of Matrix Multiplication:

1. Pure Python CPU Implementation Using for loops and arrays
2. Numpy Built in MatMul
3. PyTorch Matmul
4. OpenAI's Triton naive MatMul (1 pid per output element)
5. Triton Blocked MatMul (1 pid per "block" of output elements)
6. Nvidia CUDA C++ Naive MatMul (1 thread per output element)
7. Nvidia CUDA C++ tiled MatMul (1 thread per output element, but groups threads into localized "blocks" that cooperatively load tiles of input matrices to increase gpu cache usage and decrease global memory traffic)

CPU timings and synchronized GPU warm-ups use `time.perf_counter()`.
GPU samples use `triton.testing.do_bench_cudagraph()`.

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

- `Data/benchmark_results.csv` — summary statistics for each implementation.
- `Data/benchmark_raw_logs.csv` — warm-up and individual iteration timings.

## Project layout

- `Matmul Implementations/python_matmul.py` — pure Python implementation.
- `Matmul Implementations/Kernels/` — Triton kernels, CUDA C++ kernels, and the CUDA benchmark wrapper.
- `Matmul Implementations/Libraries/` — NumPy and PyTorch implementations.
- `Data/` — benchmark summary and raw timing CSV files.
- `Visuals/` — plotting script, logos, and generated figures.

## CUDA Library Path Note

The CUDA extension wrapper supplies Conda-specific linker flags only when `CONDA_PREFIX` is set. Outside Conda, it leaves `extra_ldflags` out of the extension loader call so PyTorch uses its defaults:

```python
from pathlib import Path
import os

conda_prefix = os.environ.get("CONDA_PREFIX")
if conda_prefix:
    cuda_lib_dir = Path(conda_prefix) / "Library" / "lib"
    cuda_link_options = {
        "extra_ldflags": [f"/LIBPATH:{cuda_lib_dir}", "cudart.lib"]
    }
else:
    cuda_link_options = {}
```

The wrapper expands these options into `load()` only for Conda environments:

```python
**cuda_link_options,
```

This configuration worked for my setup because CUDA Toolkit 12.6 was installed inside the `gpukernels` Conda environment, where `cudart.lib` was located in `Library/lib`. Most like trying to run this repo on your own will require tweaking the path to your cudart.lib file.

