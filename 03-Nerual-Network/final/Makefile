CC = gcc

CFLAGS = -Wall -O3 -flto -march=native -mtune=native
LDFLAGS = -lm

nl = 1
nh = 800
ne = 50
nb = 200
alpha = 0.1
nthreads = 256

CPU = cpu_no_blas/milestone3.c cpu_no_blas/utils.c

BLAS = cpu_blas/milestone3.c cpu_blas/utils.c

GPU = gpu/milestone3.cu gpu/utils.cu

GPU_BLAS = gpu_blas/milestone3.cu gpu_blas/utils.cu

TARGET = milstone3.o

cpu:
	$(CC) $(CFLAGS) -o $(TARGET) $(CPU) $(LDFLAGS)

blas:
	$(CC) $(CFLAGS) -lopenblas "-L/opt/homebrew/opt/openblas/lib" "-I/opt/homebrew/opt/openblas/include" -o $(TARGET) $(BLAS) $(LDFLAGS)

cuda:
	nvcc -arch=sm_86 -Xptxas -O3,-v -o $(TARGET) $(GPU) $(LDFLAGS)

cuda-blas:
	nvcc -arch=sm_86 -Xptxas -O3,-v -o $(TARGET) $(GPU_BLAS) $(LDFLAGS) -lcublas

test-cpu:
	./$(TARGET) $(nl) $(nh) $(ne) $(nb) $(alpha)

test-cuda:
	./$(TARGET) $(nl) $(nh) $(ne) $(nb) 0.001 $(nthreads)

clean:
	rm -f *.o