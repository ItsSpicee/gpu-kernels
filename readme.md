GPU Kernels Exploratory Analysis

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
