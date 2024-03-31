#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>

#define CSV_NAME_SIZE 22

void update_boundary(int N, double** C_pre) {
    for (int i=1; i<N+1; i++) {
        C_pre[i][0] = C_pre[i][N];
        C_pre[i][N+1] = C_pre[i][1];
        C_pre[0][i] = C_pre[N][i];
        C_pre[N+1][i] = C_pre[1][i];
    }
}

void save(char* fname, double** C, int start, int end) {
    FILE* fptr;
    fptr = fopen(fname, "w");
    for (int i=start; i<=end; i++) {
        for (int j=start; j<=end; j++) {
            fprintf(fptr, "%lf ", C[i][j]);
        }
        fprintf(fptr, "\n");
    }
    fclose(fptr);
}

int main(int argc, char** argv) {
    // Read the arguments
    int N = atoi(argv[1]);   // matrix dimension
    int NT = atoi(argv[2]);   // number of timesteps
    double L = atof(argv[3]);   // cartesian domain length
    double T = atof(argv[4]);   // total timespan
    double u = atof(argv[5]);   // x velocity scaler
    double v = atof(argv[6]);   // y velocity scaler

    // Compute other arguments
    double delta_x = L / N;
    double delta_t = T / NT;

    // Ensure parameters meet the Courant stability condition
    assert(delta_t <= delta_x / sqrt(2.0 * (u*u + v*v)));

    // Allocate C_pre and C_cur
    double** C_pre = (double**)malloc((N+2)*sizeof(double*));
    double** C_cur = (double**)malloc((N+2)*sizeof(double*));
    for (int i=0; i<N+2; i++) {
        C_pre[i] = (double*)malloc((N+2)*sizeof(double));
        C_cur[i] = (double*)malloc((N+2)*sizeof(double));
    }

    // Initialize C_pre and C_cur
    double x_0 = delta_x * floor(N / 2);
    double sigma_x = L / 4;
    double x = 0;
    for (int i=1; i<N+1; i++, x+=delta_x) {
        double y = 0;
        for (int j=1; j<N+1; j++, y+=delta_x) {
            double x_part = (x-x_0) * (x-x_0) / (2.0*sigma_x*sigma_x);
            double y_part = (y-x_0) * (y-x_0) / (2.0*sigma_x*sigma_x);
            C_pre[i][j] = exp(-(x_part + y_part));
        }
    }
    update_boundary(N, C_pre);

    // Run advecation
    for (int n=0; n<NT; n ++){
        if (n == 0) {
            char fname[CSV_NAME_SIZE] = "results/data/data_000";
            save(fname, C_pre, 2, N+1);
        } 
        for (int i=1; i<N+1; i++) {
            for (int j=1; j<N+1; j++) {
                double neighbor = C_pre[i-1][j] + C_pre[i+1][j] + C_pre[i][j-1] + C_pre[i][j+1];
                double u_part = u * (C_pre[i+1][j] - C_pre[i-1][j]);
                double v_part = v * (C_pre[i][j+1] - C_pre[i][j-1]);
                C_cur[i][j] = 1.0 / 4.0 * neighbor - delta_t / (2.0 * delta_x) * (u_part + v_part);
            }
        }

        // Update C_pre
        for (int i=1; i<N+1; i++){
            for (int j=1; j<N+1; j++) {
                C_pre[i][j] = C_cur[i][j];
            }
        }
        update_boundary(N, C_pre);

        // Save intermediate data at some intervals
        if (n == floor(NT/2)) {
            char fname[CSV_NAME_SIZE] = "results/data/data_001";
            save(fname, C_pre, 2, N+1);
        } else if (n == NT-1) {
            char fname[CSV_NAME_SIZE] = "results/data/data_002";
            save(fname, C_pre, 2, N+1);
        }
    }

    // Free the memory for C_pre and C_cur
    for (int i=0; i<N+2; i++) {
        free(C_pre[i]);
        free(C_cur[i]);
    }
    free(C_pre);
    free(C_cur);
}