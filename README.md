# GPU101-Project-SYMGS

[![Politecnico di Milano](https://img.shields.io/badge/University-Politecnico%20di%20Milano-blue)](https://www.polimi.it/)
[![Academic Year](https://img.shields.io/badge/Academic%20Year-2022%2F2023-green)](https://github.com/SimoneMessina0/GPU101-Project-SYMGS)
[![CUDA](https://img.shields.io/badge/Language-CUDA-brightgreen)](https://github.com/SimoneMessina0/GPU101-Project-SYMGS)

This repository contains the final project implementation for the **GPU 101** course at Politecnico di Milano. The project demonstrates advanced GPU programming concepts through the implementation of the Symmetric Gauss-Seidel (SYMGS) algorithm using CUDA.

## 📋 Table of Contents

- [Project Overview](#project-overview)
- [Algorithm Description](#algorithm-description)
- [Implementation Features](#implementation-features)
- [Getting Started](#getting-started)
- [Usage](#usage)
- [Technologies Used](#technologies-used)
- [Performance Analysis](#performance-analysis)
- [Course Information](#course-information)

## 🎯 Project Overview

This project focuses on implementing the Symmetric Gauss-Seidel (SYMGS) iterative method using CUDA for GPU acceleration. SYMGS is a symmetric variant of the classical Gauss-Seidel method, commonly used as a smoother in multigrid solvers for solving systems of linear equations of the form **Ax = b**.

The implementation demonstrates key GPU programming concepts including:
- Parallel algorithm design for inherently sequential methods
- Memory optimization techniques for GPU architecture
- Performance analysis and benchmarking

## ⚡ Algorithm Description

The Symmetric Gauss-Seidel method performs iterative solving through two phases:

1. **Forward Sweep**: Standard Gauss-Seidel iteration from first to last equation
2. **Backward Sweep**: Gauss-Seidel iteration from last to first equation

This two-phase approach maintains symmetry properties of the original matrix, which is crucial for convergence guarantees in multigrid methods.

## 🚀 Implementation Features

### How the GPU version works
- **One thread per row**: each thread computes one row of the forward (`fsweep`) or backward (`bsweep`) sweep.
- **Dependency flags**: a `done[]` flag per row says whether its new value is ready. A row whose dependency is not ready yet gives up, and the host launches another pass until every row is done.
- **Memory ordering**: `__threadfence()` on both sides of the flag handoff and L2 reads (`__ldcg`) guarantee that a row never uses a stale value written by another block.
- **Pass-id flag**: each launch writes its own pass number to the "another pass needed" flag, so the flag never has to be reset between passes.
- **Same results as the CPU**: the kernels follow the same order of operations as the reference `symgs_csr_sw`. Built with `-fmad=false`, the GPU result is bit-identical to the CPU one; with FMA (default) the difference is within floating-point rounding.

The program checks the GPU result against the CPU reference on every run and prints PASS/FAIL.

## 🔧 Getting Started

Requirements: an NVIDIA GPU, the CUDA Toolkit, and Linux or WSL2 (the code uses `sys/time.h`, which does not exist with the Windows MSVC compiler).

```bash
nvcc -O3 -arch=sm_52 -o symgs SYMGS/symgs_main.cu
```

Change `-arch` to match your GPU (`sm_52` is Maxwell, e.g. GTX 980 Ti).

## ▶️ Usage

```bash
./symgs path/to/matrix.mtx
```

The matrix file starts with a `rows cols nonzeros` header followed by one `row col value` line per nonzero (1-based indices). Every row must have a nonzero diagonal entry. The program prints the correctness check, the number of passes per sweep, and the CPU and GPU times. The exit code is 0 on PASS and 1 on FAIL.

## 🛠️ Technologies Used

- **Programming Language**: CUDA C/C++
- **GPU Platform**: NVIDIA CUDA-enabled GPUs
- **Development Environment**: NVIDIA CUDA Toolkit
- **Compiler**: nvcc (NVIDIA CUDA Compiler)

## 📊 Performance Analysis

Measured on `kmer_V4a.mtx` (51,813,503 rows, 103,565,680 nonzeros, lower triangle only) with a GTX 980 Ti, CUDA 12.9 on WSL2:

| Version | Time | Speedup vs CPU |
|---|---|---|
| CPU reference (`symgs_csr_sw`) | 777 ms | 1x |
| Original GPU version | 135 ms | ~5.8x |
| **Current GPU version** | **111 ms** | **~7.0x** |

GPU times are for one full SYMGS (forward + backward sweep) after a warm-up run, timed with CUDA events. File loading is not included.

Where the time goes (per-pass measurement, 256 threads per block):
- The first forward pass completes 99.5% of the rows (14 ms). The remaining ~72 passes finish the last 268k rows, but each one still launches a thread for all 51.8M rows (~0.85 ms each, ~63 ms total).
- The backward sweep needs a single pass (10 ms), since the matrix has no entries above the diagonal.
- Launch and flag-readback gaps between passes add ~14.5 ms in total.
- 128 threads per block was the fastest and most stable block size (93 ms vs ~95–100 ms with 256).

## 🧭 Known Limitations and Next Steps

- **Worklist**: launching only the rows that failed in the previous pass, instead of all rows, would remove most of the ~63 ms spent on later passes.
- **Block size**: switching from 256 to 128 threads per block.
- **Loop on the GPU**: a CUDA Graph conditional node or a persistent kernel would remove the ~14.5 ms of launch gaps.
- **Level scheduling**: this matrix has only 91 dependency levels, so processing one level per launch without any flags is another option.
- **Input parsing**: `read_matrix` assumes diagonal entries appear in row order, and the reference matrix file has one entry fewer than its header (handled in `main`).

## 🎓 Course Information

**Course**: GPU 101  
**Institution**: Politecnico di Milano  
**Author**: Simone Francesco Messina  
**Academic Year**: 2022/2023  

### Learning Objectives
- GPU architecture fundamentals
- CUDA programming model and optimization
- Parallel algorithm design principles
- Performance analysis and profiling techniques

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.
Note: This project was developed for educational purposes as part of the GPU 101 course at Politecnico di Milano.

---

*For questions about this project, please contact the author.*
