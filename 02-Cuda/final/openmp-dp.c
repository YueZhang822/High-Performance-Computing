#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#include "random.h"
#include "omp.h"

#define MAX(X, Y) (((X) > (Y)) ? (X) : (Y))

struct vector {
    double x, y, z;
} typedef vector;

vector vector_sub(vector v1, vector v2) {
    vector vector_res = {v1.x - v2.x, v1.y - v2.y, v1.z - v2.z};
    return vector_res;
}

vector scalar_mult(double scalar, vector v) {
    vector vector_res = {scalar * v.x, scalar * v.y, scalar * v.z};
    return vector_res;
}

double dot_product(vector v1, vector v2) {
    return v1.x * v2.x + v1.y * v2.y + v1.z * v2.z;
}

double length(vector v) {
    return sqrt(dot_product(v, v));
}

vector normalize(vector v) {
    return scalar_mult(1.0 / length(v), v);
}

double check_validity(vector V, vector W, vector C, double Wmax, double R) {
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

double uniform_distribution(double min, double max) {
    return min + (double)genrand_float32_full() * (max - min);
}

vector sample_v() {
    double phi = uniform_distribution(0.0, M_PI);
    double cos_theta = uniform_distribution(-1.0, 1.0);
    double sin_theta = sqrt(1.0 - (cos_theta * cos_theta));
    vector V = {sin_theta * cos(phi), sin_theta * sin(phi), cos_theta};
    return V;
}

void save_matrix(char* fname, double** G, int n) {
    FILE* fptr = fopen(fname, "w");
    for (int i=0; i<n; i++) {
        for (int j=0; j<n; j++) {
            fprintf(fptr, "%lf ", G[i][j]);
        }
        fprintf(fptr, "\n");
    }
    fclose(fptr);
}

int main(int argc, char** argv) {
    // Reads command line arguments
    int n_ray = atoi(argv[1]);
    int n_grid = atoi(argv[2]);
    int n_threads = atoi(argv[5]);

    // Initialize default arguments
    vector L = {4, 4, -1};
    double Wy = 2.0;
    double Wmax = 2.0;
    vector C = {0, 12, 0};
    double R = 6.0;
    long long* count = malloc(n_threads*sizeof(long long));

    omp_set_num_threads(n_threads);
    printf("Number of threads: %d\n", n_threads);

    // Allocate and initilize grid G to represent the window
    double** G = (double**) malloc(n_grid*sizeof(double*));
    for (int i=0; i<n_grid; i++) {
        G[i] = (double*) calloc(n_grid, sizeof(double));
    }

    // Set the random seed
    time_t t;
    init_genrand((unsigned)time(&t));

    // Measure running time
    double start = omp_get_wtime();

    #pragma omp parallel for default(none) shared(G, n_ray) firstprivate(L, C, R, Wy, Wmax, n_grid, count)
    for (int n=0; n<n_ray; n++) {
        int i, j;
        double t, b;
        vector V, W, I, N, S;
        double ray_validity = -1.0;
        int thread_count = 0;

        while (ray_validity < 0.0) {
            V = sample_v();
            W = scalar_mult(Wy / V.y, V);
            ray_validity = check_validity(V, W, C, Wmax, R);
            thread_count += 2;
        }
        t = dot_product(V, C) - sqrt(ray_validity);
        I = scalar_mult(t, V);
        N = normalize(vector_sub(I, C));
        S = normalize(vector_sub(L, I));
        b = MAX(0.0, dot_product(S, N));
        i = W.x / (Wmax / (n_grid / 2)) + n_grid / 2;
        j = W.z / (Wmax / (n_grid / 2)) + n_grid / 2;
        G[n_grid - 1 - i][j] += b;
        int tid = omp_get_thread_num();
        count[tid] += thread_count;
    }

    long long total_count = 0;
    for (int i = 0; i < n_threads; i++) {
        total_count += count[i];
    }

    // Measure running time
    double end = omp_get_wtime();
    double time_used = end - start;
    printf("Time elapsed: %lf(s)\n", time_used);
    printf("Number of samples: %lld\n", total_count);

    // Save the results to file
    char filename[] = "results/openmp-dp/res.txt";
    save_matrix(filename, G, n_grid);

    // Clean up the memory
    for (int i=0; i<n_grid; i++) {
        free(G[i]);
    }
    free(G);
    free(count);
}