#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <omp.h>

#define CSV_NAME_SIZE 22

void update_boundary(int N, double** C_pre) {
    for (int i=2; i<N+2; i++) {
        C_pre[i][0] = C_pre[i][N];
        C_pre[i][1] = C_pre[i][N+1];
        C_pre[i][N+2] = C_pre[i][2];
        C_pre[i][N+3] = C_pre[i][3];
        C_pre[0][i] = C_pre[N][i];
        C_pre[1][i] = C_pre[N+1][i];
        C_pre[N+2][i] = C_pre[2][i];
        C_pre[N+3][i] = C_pre[3][i];
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

void update(int N, int n, int NT, double** C_pre, double** C_cur) {
    // Update C_pre
        for (int i=2; i<N+2; i++){
            for (int j=2; j<N+2; j++) {
                C_pre[i][j] = C_cur[i][j];
            }
        }
        update_boundary(N, C_pre);
}

void lax_serial(int N, int n, int NT, double** C_pre, double** C_cur, double u, double v, double delta_x, double delta_t) {
    for (int i=2; i<N+2; i++) {
        for (int j=2; j<N+2; j++) {
            double neighbor = C_pre[i-1][j] + C_pre[i+1][j] + C_pre[i][j-1] + C_pre[i][j+1];
            double u_part = u * (C_pre[i+1][j] - C_pre[i-1][j]);
            double v_part = v * (C_pre[i][j+1] - C_pre[i][j-1]);
            C_cur[i][j] = 1.0 / 4.0 * neighbor - delta_t / (2.0 * delta_x) * (u_part + v_part);
        }
    }
    update(N, n, NT, C_pre, C_cur);
}

void lax_parallel(int N, int n, int NT, double** C_pre, double** C_cur, double u, double v, double delta_x, double delta_t) {
    #pragma omp parallel for default(none) shared(C_pre, C_cur) firstprivate(N, delta_x, delta_t, u, v)
    for (int i=2; i<N+2; i++) {
        for (int j=2; j<N+2; j++) {
            double neighbor = C_pre[i-1][j] + C_pre[i+1][j] + C_pre[i][j-1] + C_pre[i][j+1];
            double u_part = u * (C_pre[i+1][j] - C_pre[i-1][j]);
            double v_part = v * (C_pre[i][j+1] - C_pre[i][j-1]);
            C_cur[i][j] = 1.0 / 4.0 * neighbor - delta_t / (2.0 * delta_x) * (u_part + v_part);
        }
    }
    update(N, n, NT, C_pre, C_cur);
}

void first_serial(int N, int n, int NT, double** C_pre, double** C_cur, double u, double v, double delta_x, double delta_t){
    for (int i=2; i<N+2; i++){
        for (int j=2; j<N+2; j++){
            double ufactor;
            double vfactor;
            if (u>0 && v>0){
                ufactor = (C_pre[i][j] - C_pre[i-1][j]) / delta_x;
                vfactor = (C_pre[i][j] - C_pre[i][j-1]) / delta_x;
            } else {
                ufactor = (C_pre[i+1][j] - C_pre[i][j]) / delta_x;
                vfactor = (C_pre[i][j+1] - C_pre[i][j]) / delta_x;
            }
            C_cur[i][j] = C_pre[i][j] - delta_t * (u * ufactor + v * vfactor);
        }
    }
    update(N, n, NT, C_pre, C_cur);
}

void first_parallel(int N, int n, int NT, double** C_pre, double** C_cur, double u, double v, double delta_x, double delta_t){
    #pragma omp parallel for default(none) shared(C_pre, C_cur) firstprivate(N, delta_x, delta_t, u, v)
    for (int i=2; i<N+2; i++){
        for (int j=2; j<N+2; j++){
            double ufactor;
            double vfactor;
            if (u>0 && v>0){
                ufactor = (C_pre[i][j] - C_pre[i-1][j]) / delta_x;
                vfactor = (C_pre[i][j] - C_pre[i][j-1]) / delta_x;
            } else {
                ufactor = (C_pre[i+1][j] - C_pre[i][j]) / delta_x;
                vfactor = (C_pre[i][j+1] - C_pre[i][j]) / delta_x;
            }
            C_cur[i][j] = C_pre[i][j] - delta_t * (u * ufactor + v * vfactor);
        }
    }
    update(N, n, NT, C_pre, C_cur);
}

void second_serial(int N, int n, int NT, double** C_pre, double** C_cur, double u, double v, double delta_x, double delta_t){
    for (int i=2; i<N+2; i++){
        for (int j=2; j<N+2; j++){
            double ufactor;
            double vfactor;
            if (u>0 && v>0){
                ufactor = u * (C_pre[i][j] * 3.0 - C_pre[i-1][j] * 4.0 + C_pre[i-2][j]);
                vfactor = v * (C_pre[i][j] * 3.0 - C_pre[i][j-1] * 4.0 + C_pre[i][j-2]);
            } else{
                ufactor = (-1.0 * C_pre[i+2][j] + 4.0 * C_pre[i+1][j] - 3.0 * C_pre[i][j]) / (delta_x * 2.0);
                vfactor = (-1.0 * C_pre[i][j+2] + 4.0 * C_pre[i][j+1] - 3.0 * C_pre[i][j]) / (delta_x * 2.0);
            }
            double d = delta_t / (delta_x * 2.0);
            C_cur[i][j] = C_pre[i][j] - d * (ufactor + vfactor);
        }
    }
    update(N, n, NT, C_pre, C_cur);
}

void second_parallel(int N, int n, int NT, double** C_pre, double** C_cur, double u, double v, double delta_x, double delta_t){
    #pragma omp parallel for default(none) shared(C_pre, C_cur) firstprivate(N, delta_x, delta_t, u, v)
    for (int i=2; i<N+2; i++){
        for (int j=2; j<N+2; j++){
            double ufactor;
            double vfactor;
            if (u>0 && v>0){
                ufactor = (3 * C_pre[i][j] - 4 * C_pre[i-1][j] + C_pre[i-2][j]) / (delta_x * 2);
                vfactor = (3 * C_pre[i][j] - 4 * C_pre[i][j-1] + C_pre[i][j-2]) / (delta_x * 2);
            } else{
                ufactor = (-1 * C_pre[i+2][j] + 4 * C_pre[i+1][j] - 3 * C_pre[i][j]) / (delta_x * 2);
                vfactor = (-1 * C_pre[i][j+2] + 4 * C_pre[i][j+1] - 3 * C_pre[i][j]) / (delta_x * 2);
            }
            C_cur[i][j] = -1 * delta_t * (u * ufactor + v * vfactor) + C_pre[i][j];
        }
    }
    update(N, n, NT, C_pre, C_cur);
}

int main(int argc, char** argv) {
    // The number of input has to be 7
    if (argc != 7) {
        printf("%i\n", argc);
        printf("Usage: %s <N> <L> <T> <u> <v> <num_threads>\n", argv[0]);
        return 1;
    }

    // Read the arguments
    int N = atoi(argv[1]);      // matrix dimension
    double L = atof(argv[2]);   // cartesian domain length
    double T = atof(argv[3]);   // total timespan
    double u = atof(argv[4]);   // x velocity scaler
    double v = atof(argv[5]);   // y velocity scaler
    int num_threads = atoi(argv[6]);   // number of threads

    // Compute other arguments
    double delta_x = L / (N - 1);
    double delta_y = L / (N - 1);
    double delta_t = 0.5 * delta_x / sqrt(u*u + v*v);
    int NT = T / delta_t;   // number of timesteps

    double t=0, t1, t2;

    // Debug
    printf("N=%i L=%f T=%f u=%f v=%f thread=%i\n", N, L, T, u, v, num_threads);
    printf("delta_x=%f delta_y=%f delta_t=%f NT=%i\n", delta_x, delta_y, delta_t, NT);

    // Ensure parameters meet the Courant stability condition
    assert(delta_t <= delta_x / sqrt(2.0 * (u*u + v*v)));
    
    omp_set_num_threads(num_threads);

    // Allocate C_pre and C_cur
    double** C_pre = (double**)malloc((N+4)*sizeof(double*));
    double** C_cur = (double**)malloc((N+4)*sizeof(double*));
    for (int i=0; i<N+4; i++) {
        C_pre[i] = (double*)malloc((N+4)*sizeof(double));
        C_cur[i] = (double*)malloc((N+4)*sizeof(double));
    }

    // Initialize C_pre and C_cur
    double x = -L / 2;
    for (int i=2; i<N+2; i++, x+=delta_x) {
        double y = -L / 2;
        for (int j=2; j<N+2; j++, y+=delta_y) {
            C_pre[i][j] = exp(-100 * (x*x + y*y));
        }
    }
    update_boundary(N, C_pre);

    // Run advecation
    for (int n=0; n<NT; n ++){
        if (n == 0) {
            char fname[CSV_NAME_SIZE] = "results/data/data_000";
            save(fname, C_pre, 2, N+1);
        } 
        t1 = omp_get_wtime();
        #ifdef LAX
        #ifdef SERIAL 
            lax_serial(N, n, NT, C_pre, C_cur, u, v, delta_x, delta_t);
        #else
            lax_parallel(N, n, NT, C_pre, C_cur, u, v, delta_x, delta_t);
        #endif
        #endif

        #ifdef FIRST 
        #ifdef SERIAL
            first_serial(N, n, NT, C_pre, C_cur, u, v, delta_x, delta_t); 
        #else
            first_parallel(N, n, NT, C_pre, C_cur, u, v, delta_x, delta_t);
        #endif
        #endif

        #ifdef SECOND
        #ifdef SERIAL
            second_serial(N, n, NT, C_pre, C_cur, u, v, delta_x, delta_t);
        #else
            second_parallel(N, n, NT, C_pre, C_cur, u, v, delta_x, delta_t);
        #endif
        #endif

        t2 = omp_get_wtime();
        t += t2 - t1;

        // Save intermediate data at some intervals
        if (n == floor(NT/2)) {
            char fname[CSV_NAME_SIZE] = "results/data/data_001";
            save(fname, C_pre, 2, N+1);
        } else if (n == NT-1) {
            char fname[CSV_NAME_SIZE] = "results/data/data_002";
            save(fname, C_pre, 2, N+1);
        }
    }

    printf("time(s): %f\n", t);

    // Clean up memory
    for (int i=0; i<N+4; i++) {
        free(C_pre[i]);
        free(C_cur[i]);
    }
    free(C_pre);
    free(C_cur);
}