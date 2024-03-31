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
    float x, y, z;
} typedef vector;

__device__ vector vector_sub(vector v1, vector v2) {
    vector vector_res = {v1.x - v2.x, v1.y - v2.y, v1.z - v2.z};
    return vector_res;
}

__device__ vector scalar_mult(float scalar, vector v) {
    vector vector_res = {scalar * v.x, scalar * v.y, scalar * v.z};
    return vector_res;
}

__device__ float dot_product(vector v1, vector v2) {
    return v1.x * v2.x + v1.y * v2.y + v1.z * v2.z;
}

__device__ float length(vector v) {
    return sqrt(dot_product(v, v));
}

__device__ vector normalize(vector v) {
    return scalar_mult(1.0 / length(v), v);
}

__device__ float check_validity(vector V, vector W, vector C, float Wmax, float R) {
    float vc, cc, factor;
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
    float phi = curand_uniform(d_state) * M_PI;
    float cos_theta = curand_uniform(d_state) * 2.0 - 1.0;
    float sin_theta = sqrt(1.0 - (cos_theta * cos_theta));
    vector V = {sin_theta * cos(phi), sin_theta * sin(phi), cos_theta};
    return V;
}

__host__ void save_matrix(char* fname, float* G, int n) {
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
__host__ __device__ void increment_data(float* grid, int i, int j, int size, float b) {
    grid[i * size + j] += b;
}

__global__ void setup_kernel(curandState *state, unsigned long seed) {
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    curand_init(seed, id, 0, &state[id]);
}

__global__ void ray_tracing_kernel(float* G, vector L, float Wy, float Wmax, vector C, float R, int n_work, int n_grid, curandState *d_state) {
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    for (int k=0; k<n_work; k++) {
        int i, j;
        float t, b;
        vector V, W, I, N, S;
        float ray_validity = -1.0;

        while (ray_validity < 0.0) {
            V = sample_v(d_state + n);
            W = scalar_mult(Wy / V.y, V);
            ray_validity = check_validity(V, W, C, Wmax, R);
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

int main(int argc, char** argv) {
    // Reads command line arguments
    vector L = {(float)atof(argv[1]), (float)atof(argv[2]), (float)atof(argv[3])};
    float Wy = (float)atof(argv[4]);
    float Wmax = (float)atof(argv[5]);
    vector C = {(float)atof(argv[6]), (float)atof(argv[7]), (float)atof(argv[8])};
    float R = (float)atof(argv[9]);
    int n_ray = atoi(argv[10]);
    int n_grid = atoi(argv[11]);
    int nthreads = atoi(argv[12]);    // Number of threads per block
    int nblocks = atoi(argv[13]);     // Number of blocks
    int work_per_thread = ceil((float)(n_ray) / (nblocks * nthreads));
    printf("BLOCKS: %d;  THREADS: %d; WORK PER THREAD %d\n", nblocks, nthreads, work_per_thread);

    curandState *d_state;
    assert(cudaMalloc(&d_state, nthreads*nblocks*sizeof(curandState)) == cudaSuccess);

    // Allocate and initilize grid G and cuda_G
    float* G = NULL;
    assert(cudaMallocHost((void**)&G, n_grid*n_grid*sizeof(float)) == cudaSuccess);
    assert(cudaMemset(G, 0, n_grid*n_grid*sizeof(float)) == cudaSuccess);
    float* cuda_G = NULL;
    assert(cudaMalloc((void**)&cuda_G, n_grid*n_grid*sizeof(float)) == cudaSuccess);
    assert(cudaMemcpy(cuda_G, G, n_grid*n_grid*sizeof(float), cudaMemcpyHostToDevice) == cudaSuccess);

    // Measure running time
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float gpu_time;

    // Launch the kernel
    cudaEventRecord(start, 0);

    setup_kernel<<<nblocks, nthreads>>>(d_state, time(NULL));

    ray_tracing_kernel<<<nblocks, nthreads>>>(cuda_G, L, Wy, Wmax, C, R, work_per_thread, n_grid, d_state);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&gpu_time, start, stop);
    printf("Time elapsed on GPU: %f(s)\n", gpu_time / 1000.0);
    assert(cudaMemcpy(G, cuda_G, n_grid*n_grid*sizeof(float), cudaMemcpyDeviceToHost) == cudaSuccess);

    // Save the results to file
    char filename[16] = "results/res.txt";
    save_matrix(filename, G, n_grid);

    // Clean up the memory
    cudaFree(d_state);
    cudaFreeHost(G);
    cudaFree(cuda_G);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}