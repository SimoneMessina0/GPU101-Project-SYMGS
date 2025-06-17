# GPU101-Project-SYMGS

[![Politecnico di Milano](https://img.shields.io/badge/University-Politecnico%20di%20Milano-blue)](https://www.polimi.it/)
[![Academic Year](https://img.shields.io/badge/Academic%20Year-2024%2F2025-green)](https://github.com/SimoneMessina0/GPU101-Project-SYMGS)
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

### GPU Optimization Techniques
- **Red-Black Ordering**: Parallelization of the inherently sequential Gauss-Seidel method
- **Coalesced Memory Access**: Optimized memory access patterns for GPU architecture
- **Shared Memory Utilization**: Reduction of global memory accesses through local buffering
- **Thread Block Optimization**: Efficient GPU resource utilization

### Performance Capabilities
- **Significant Speedup**: GPU acceleration over sequential CPU implementations
- **Scalable Performance**: Efficient handling of large matrix systems
- **Memory Bandwidth Optimization**: Maximized utilization of GPU memory hierarchy

## 🛠️ Getting Started

### Prerequisites
- NVIDIA GPU with CUDA capability 3.0 or higher
- CUDA Toolkit (version 8.0 or later recommended)
- GCC compiler compatible with your CUDA version

### Building the Project
```bash
# Clone the repository
git clone https://github.com/SimoneMessina0/GPU101-Project-SYMGS.git
cd GPU101-Project-SYMGS

# Compile the project
nvcc -o symgs *.cu -O3 -arch=sm_60
```

*Replace `sm_60` with the appropriate compute capability for your GPU.*

## 💻 Usage

### Basic Execution
```bash
./symgs <matrix_size> <num_iterations>
```

### Example
```bash
./symgs 1024 100
```

This runs the SYMGS solver on a 1024×1024 matrix for 100 iterations.

## 🛠️ Technologies Used

- **Programming Language**: CUDA C/C++
- **GPU Platform**: NVIDIA CUDA-enabled GPUs
- **Development Environment**: NVIDIA CUDA Toolkit
- **Compiler**: nvcc (NVIDIA CUDA Compiler)

## 📊 Performance Analysis

The implementation includes capabilities for analyzing:
- **Convergence Rate**: Iterations required for solution convergence
- **GPU Utilization**: Percentage of GPU resources effectively used
- **Memory Bandwidth**: Achieved memory throughput performance
- **Speedup Metrics**: Performance comparison with CPU implementations

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
