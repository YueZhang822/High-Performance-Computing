# High Performance Computing

This repository is a collection of projects for high performance computing learning, covering practices in MPI, OPenMP, CUDA and other high performance computing techniques.

## Project 1: Parallelized Advection Solver

* Milestone 1: Implemented a serial solution for numerically estimating the 2D advection equation using the Lax method. 

* Milestone 2: Extended the existing Lax solver by adding two additional numerical methods—first and second-order upwind schemes, and parallelized all three solvers using OpenMP for multicore processing.

* Final Milestone: Implemented and tested the Lax solver under various conditions, including serial, shared memory parallel, and distributed memory parallel environments.

## Project 2: GPU Ray Tracing with CUDA

* Milestone 1: Implemented a serial version of the ray tracing algorithm to simulate the reflection of light rays from a single light source off a reflective sphere. 

* Milestone 2: Parallelized the code for multicore CPU usage with OpenMP and performed an initial port to NVIDIA GPUs using CUDA.

* Final Milestone: Completed the CUDA implementation. Conducted an evaluation and optimization of the ray tracing algorithm across various computing platforms, including different GPU models (A100, V100, RTX6000) and CPU configurations, both in serial and with OpenMP.

## Project 3: Machine Learning for Image Classification

* Milestone 1: Developed and trained a basic Feedforward Neural Network (FNN) for digit recognition. This initial phase involved creating a CPU-based implementation, setting the stage for future GPU optimization.

* Milestone 2: Improved the model by transitioning to a more robust configuration (details skipped). Introduced a baseline GPU version. Evaluated performance by comparing execution time, grind rate, and accuracy across various setups: CPU without BLAS, CPU with BLAS, and the baseline GPU implementation.

* Final: Enhanced the GPU version of the machine learning model by incorporating CUDA's BLAS library (CuBLAS), aiming to boost performance significantly.