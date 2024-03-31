#include "mpi.h"
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>

#define CSV_NAME_SIZE 40
#define MPI_REQUEST_NUM 8
#define PADDING_SZIE 2
#define min(X, Y) (((X) < (Y)) ? (X) : (Y))
#define max(X, Y) (((X) > (Y)) ? (X) : (Y))
#define idx(C, i, j, N_x) (C[(i)*(N_x+PADDING_SZIE)+(j)])

void update_boundary(int N_x, int N_y, double** C_pre, int up, int down, int left, int right, double* sbuff_left, double* sbuff_right, double* rbuff_left, double* rbuff_right) {
    for (int i=1;i<N_y+1;i++) {
        sbuff_left[i] = C_pre[i][1];
        sbuff_right[i] = C_pre[i][N_x];
    }
    MPI_Request request[MPI_REQUEST_NUM] = {};
    MPI_Isend(C_pre[1], N_x+1, MPI_DOUBLE, up, 1, MPI_COMM_WORLD, request);
    MPI_Isend(C_pre[N_y], N_x+1, MPI_DOUBLE, down, 2, MPI_COMM_WORLD, request+1);
    MPI_Isend(sbuff_left, N_y, MPI_DOUBLE, left, 3, MPI_COMM_WORLD, request+2);
    MPI_Isend(sbuff_right, N_y, MPI_DOUBLE, right, 4, MPI_COMM_WORLD, request+3);

    MPI_Irecv(C_pre[0], N_x+1, MPI_DOUBLE, up, 2, MPI_COMM_WORLD, request+4);
    MPI_Irecv(C_pre[N_y+1], N_x+1, MPI_DOUBLE, down, 1, MPI_COMM_WORLD, request+5);
    MPI_Irecv(rbuff_left, N_y, MPI_DOUBLE, left, 4, MPI_COMM_WORLD, request+6);
    MPI_Irecv(rbuff_right, N_y, MPI_DOUBLE, right, 3, MPI_COMM_WORLD, request+7);

    for (int i=0;i<MPI_REQUEST_NUM;i++){
        MPI_Wait(request+i, MPI_STATUS_IGNORE);
    }

    for (int i=1;i<N_y+1;i++){
        C_pre[i][0] = rbuff_left[i];
        C_pre[i][N_x+1] = rbuff_right[i];
    }
}

void save(char* fname, double** C, int X_start, int Y_start, int X_end, int Y_end) {
    FILE* fptr;
    fptr = fopen(fname, "w");
    for (int i=Y_start; i<Y_end; i++) {
        for (int j=X_start; j<X_end; j++) {
            fprintf(fptr, "%lf ", C[i][j]);
        }
        fprintf(fptr, "\n");
    }
    fclose(fptr);
}

int main(int argc, char** argv) {
    int nprocs;
    int mype;
    MPI_Status stat;
    
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, &mype);

    int NT, N_x, N_y, m, n;
    int length_x, length_y;
    double L, delta_t, delta_x, sqrt2;
    int strt, end;
    int num_threads;

    if (mype == 0){
        // Read the arguments
        int N = atoi(argv[1]);
        L = atof(argv[2]);
        double T = atof(argv[3]);
        delta_t = atof(argv[4]);
        m = atoi(argv[5]);
        n = atoi(argv[6]);
        // Compute other arguments
        NT = T / delta_t;
        N_x = N / n;
        N_y = N / m;
        delta_x = L / (N - 1);
        length_x = N / n;
        length_y = N / m;
        strt = (int)(0.4 * L / delta_x);
        end = (int)(0.6 * L / delta_x);
        num_threads = atoi(argv[7]);
    }

    int param_int[] = {NT, N_x, N_y, m, n, length_x, length_y, strt, end, num_threads};
    double param_double[] = {L, delta_t, delta_x};

    MPI_Bcast(param_int, 10, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(param_double, 3, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    NT = param_int[0];
    N_x = param_int[1];
    N_y = param_int[2];
    m = param_int[3];
    n = param_int[4];
    length_x = param_int[5];
    length_y = param_int[6];
    L = param_double[0];
    delta_t = param_double[1];
    delta_x = param_double[2];
    strt = param_int[7];
    end = param_int[8];
    num_threads = param_int[9];

    omp_set_num_threads(num_threads);

    sqrt2 = sqrt(2);

    int col = mype % n;
    int row = mype / n;
    int X_start = length_x * col;
    int Y_start = length_y * row;
    int X_end = length_x * (col + 1) - 1;
    int Y_end = length_y * (row + 1) - 1;

    int up = ((row - 1 + m) % m) * n + col;
    int down = ((row + 1) % m) * n + col;
    int left = row * n + ((col - 1 + n) % n);
    int right = row * n + ((col + 1) % n);

    double t=0, t1, t2;

    // printf("rank=%d up=%d down=%d left=%d right=%d\n", mype, up, down, left, right);
    // printf("rank=%d Y_start=%d Y_end=%d X_start=%d, X_end=%d N_x=%d, N_y=%d\n", mype, Y_start, Y_end, X_start, X_end, N_x, N_y);
    // printf("rank=%d end=%d Y_end=%d min=%d N_x=%d, N_y=%d\n", mype, end, Y_end, min(end, Y_end), N_x, N_y);

    double* sbuff_left = (double*) malloc(N_y * sizeof(double));
    double* sbuff_right = (double*) malloc(N_y * sizeof(double));

    double* rbuff_left = (double*) malloc(N_y * sizeof(double));
    double* rbuff_right = (double*) malloc(N_y * sizeof(double));

    // Allocate C_pre and C_cur
    int l = max(N_x, N_y) + PADDING_SZIE;
    double** C_pre = (double**)malloc(l*sizeof(double*));
    double* C_cur = (double*)malloc(l*l*sizeof(double));
    for (int i=0; i<l; i++) {
        C_pre[i] = (double*)malloc(l*sizeof(double));
    }

    // Initialize C_pre and C_cur
    for (int i=1; i<N_y+1; i++) {
        for (int j=1; j<N_x+1; j++) {
            C_pre[i][j] = 0;
        }
    }

    for (int i=max(strt, Y_start)-Y_start+1; i<=min(end, Y_end)-Y_start+1; i++) {
        for (int j=1; j<N_x+1; j++) {
            C_pre[i][j] = 1;
        }
    }
    update_boundary(N_x, N_y, C_pre, up, down, left, right, sbuff_left, sbuff_right, rbuff_left, rbuff_right);

    // Run advecation
    for (int n=0; n<=NT; n ++){
        if (n == 0) {
            char fname[CSV_NAME_SIZE];
            sprintf(fname, "results/data/data_%d_000", mype);
            save(fname, C_pre, 1, 1, N_x+1, N_y+1);
        }
        MPI_Barrier(MPI_COMM_WORLD);
        t1 = MPI_Wtime();
        #pragma omp parallel for default(none) shared(C_cur) firstprivate(C_pre, N_y, N_x, delta_t, delta_x, L, sqrt2, X_start, Y_start)
        for (int i=1; i<N_y+1; i++) {
            double y = Y_start * delta_x - L / 2.0 + (i-1) * delta_x;
            double u = sqrt2 * y;
            double x = X_start * delta_x - L / 2.0;
            for (int j=1; j<N_x+1; j++, x+=delta_x) {
                double v = -sqrt2 * x;
                double neighbor = C_pre[i-1][j] + C_pre[i+1][j] + C_pre[i][j-1] + C_pre[i][j+1];
                double u_part = v * (C_pre[i+1][j] - C_pre[i-1][j]);
                double v_part = u * (C_pre[i][j+1] - C_pre[i][j-1]);
                idx(C_cur, i, j, N_x) = 1.0 / 4.0 * neighbor - delta_t / (2.0 * delta_x) * (u_part + v_part);
            }
        }
        

        // Update C_pre
        for (int i=1; i<N_y+1; i++){
            for (int j=1; j<N_x+1; j++) {
                C_pre[i][j] = idx(C_cur, i, j, N_x);
            }
        }
        update_boundary(N_x, N_y, C_pre, up, down, left, right, sbuff_left, sbuff_right, rbuff_left, rbuff_right);

        MPI_Barrier(MPI_COMM_WORLD);
        t2 = MPI_Wtime();
        t += t2 - t1;

        // Save intermediate data at some intervals
        if (n == floor(NT/2)) {
            char fname[CSV_NAME_SIZE];
            sprintf(fname, "results/data/data_%d_001", mype);
            save(fname, C_pre, 1, 1, N_x, N_y);
        } else if (n == NT-1) {
            char fname[CSV_NAME_SIZE];
            sprintf(fname, "results/data/data_%d_002", mype);
            save(fname, C_pre, 1, 1, N_x, N_y);
        }
    }
    if (mype == 0) {
        printf("time(s): %f\n", t);
    }

    // Free the memory for C_pre and C_cur
    for (int i=0; i<l; i++) {
        free(C_pre[i]);
    }
    free(C_pre);
    free(C_cur);
    free(sbuff_left);
    free(sbuff_right);
    free(rbuff_left);
    free(rbuff_right);

    MPI_Finalize();
}
