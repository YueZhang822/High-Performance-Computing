#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <assert.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>

#define MAX(X, Y) (((X) > (Y)) ? (X) : (Y))
#define MIN(X, Y) (((X) < (Y)) ? (X) : (Y))

struct vector {
    double x, y, z;
} typedef vector;

__device__ vector vector_sub(vector v1, vector v2) {
    vector vector_res = {v1.x - v2.x, v1.y - v2.y, v1.z - v2.z};
    return vector_res;
}

__device__ vector scalar_mult(double scalar, vector v) {
    vector vector_res = {scalar * v.x, scalar * v.y, scalar * v.z};
    return vector_res;
}

__device__ double dot_product(vector v1, vector v2) {
    return v1.x * v2.x + v1.y * v2.y + v1.z * v2.z;
}

__device__ double length(vector v) {
    return sqrt(dot_product(v, v));
}

__device__ vector normalize(vector v) {
    return scalar_mult(1.0 / length(v), v);
}

__device__ double check_validity(vector V, vector W, vector C, double Wmax, double R) {
    double vc, cc, factor;
    if (fabs(W.x) >= Wmax || fabs(W.z) >= Wmax) {
        return -1.0;
    }
    vc = dot_product(V, C);
    cc = dot_product(C, C);
    factor = vc * vc + R * R - cc;
    if (factor <= 0) {
        return -1.0;
    }
    return factor;
}

__device__ vector sample_v(curandState *d_state) {
    double phi = curand_uniform(d_state) * M_PI;
    double cos_theta = curand_uniform(d_state) * 2.0 - 1.0;
    double sin_theta = sqrt(1.0 - (cos_theta * cos_theta));
    vector V = {sin_theta * cos(phi), sin_theta * sin(phi), cos_theta};
    return V;
}

__host__ void save_matrix(char* fname, double* G, int n) {
    FILE* fptr = fopen(fname, "w");
    for (int i=0; i<n; i++) {
        for (int j=0; j<n; j++) {
            fprintf(fptr, "%lf ", G[i * n + j]);
        }
        fprintf(fptr, "\n");
    }
    fclose(fptr);
}

/* Add up the contribution of the rays to the grid */
__host__ __device__ void increment_data(double* grid, int i, int j, int size, double b) {
    grid[i * size + j] += b;
}

__global__ void setup_kernel(curandState *state, unsigned long seed) {
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    curand_init(seed, id, 0, &state[id]);
}

__global__ void ray_tracing_kernel(double* G, vector L, double Wy, double Wmax, vector C, double R, int n_work, int n_grid, curandState *d_state, int* count) {
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    for (int k=0; k<n_work; k++) {
        int i, j;
        double t, b;
        vector V, W, I, N, S;
        double ray_validity = -1.0;

        while (ray_validity < 0.0) {
            V = sample_v(d_state + n);
            W = scalar_mult(Wy / V.y, V);
            ray_validity = check_validity(V, W, C, Wmax, R);
            count[n] += 2;
        }
        t = dot_product(V, C) - sqrt(ray_validity);
        I = scalar_mult(t, V);
        N = normalize(vector_sub(I, C));
        S = normalize(vector_sub(L, I));
        b = MAX(0.0, dot_product(S, N));
        i = W.x / (Wmax / (n_grid / 2)) + n_grid / 2;
        j = W.z / (Wmax / (n_grid / 2)) + n_grid / 2;
        increment_data(G, n_grid - 1 - i, j, n_grid, b);
    }
}

void save(double* G, int n_grid) {
    char filename[] = "results/gpu-dp/res.txt";
    save_matrix(filename, G, n_grid);
}

int main(int argc, char** argv) {
    // Measure running time
    clock_t t_start, t_end, t_total;
    t_start = clock();

    // Reads command line arguments
    int n_ray = atoi(argv[1]);
    int n_grid = atoi(argv[2]);
    int nblocks = atoi(argv[3]);
    int nthreads = atoi(argv[4]);
    int work_per_thread = ceil((double)(n_ray) / (nblocks * nthreads));
    assert(nthreads <= 1024);

    // Initialize default arguments
    vector L = {4, 4, -1};
    double Wy = 2.0;
    double Wmax = 2.0;
    vector C = {0, 12, 0};
    double R = 6.0;
    int* count = NULL;
    assert(cudaMalloc((void**)&count, nblocks*nthreads*sizeof(int)) == cudaSuccess);
    assert(cudaMemset(count, 0, nblocks*nthreads*sizeof(int)) == cudaSuccess);
    int* host_count = NULL;
    assert(cudaMallocHost((void**)&host_count, nblocks*nthreads*sizeof(int)) == cudaSuccess);

    // printf("BLOCKS: %d;  THREADS: %d; WORK PER THREAD %d\n", nblocks, nthreads, work_per_thread);

    curandState *d_state;
    assert(cudaMalloc(&d_state, nthreads*nblocks*sizeof(curandState)) == cudaSuccess);

    // Allocate and initilize grid G and cuda_G
    double* G = NULL;
    assert(cudaMallocHost((void**)&G, n_grid*n_grid*sizeof(double)) == cudaSuccess);
    assert(cudaMemset(G, 0, n_grid*n_grid*sizeof(double)) == cudaSuccess);
    double* cuda_G = NULL;
    assert(cudaMalloc((void**)&cuda_G, n_grid*n_grid*sizeof(double)) == cudaSuccess);
    assert(cudaMemcpy(cuda_G, G, n_grid*n_grid*sizeof(double), cudaMemcpyHostToDevice) == cudaSuccess);

    // Measure running time
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float gpu_time;

    // Launch the kernel
    cudaEventRecord(start, 0);

    setup_kernel<<<nblocks, nthreads>>>(d_state, time(NULL));

    ray_tracing_kernel<<<nblocks, nthreads>>>(cuda_G, L, Wy, Wmax, C, R, work_per_thread, n_grid, d_state, count);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&gpu_time, start, stop);

    assert(cudaMemcpy(G, cuda_G, n_grid*n_grid*sizeof(double), cudaMemcpyDeviceToHost) == cudaSuccess);
    assert(cudaMemcpy(host_count, count, nblocks*nthreads*sizeof(int), cudaMemcpyDeviceToHost) == cudaSuccess);

    long long total_count = 0;
    for (int i = 0; i < nblocks*nthreads; i++) {
        total_count += host_count[i];
    }

    // Save the results to file
    t_end = clock();
    t_total = t_end - t_start;

    save(G, n_grid);

    t_start = clock();

    // Clean up the memory
    cudaFree(d_state);
    cudaFreeHost(G);
    cudaFree(cuda_G);
    cudaFree(count);
    cudaFree(host_count);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    t_end = clock();
    float cpu_time_used = ((float) (t_total + t_end - t_start)) / CLOCKS_PER_SEC;
    printf("Total time elapsed: %lf(s)\n", cpu_time_used);
    printf("Time elapsed on GPU: %f(s)\n", gpu_time / 1000.0);
    printf("Number of samples: %lld\n", total_count);
}