CC = gcc

CFLAGS = -Wall -O3 -flto -march=native -mtune=native
LDFLAGS = -lm

nl = 1
nh = 800
ne = 5
nb = 256
alpha = 0.01
nthreads = 256

CPU = cpu_no_blas/milestone2.c cpu_no_blas/utils.c

BLAS = cpu_blas/milestone2.c cpu_blas/utils.c

GPU = gpu/milestone2.cu gpu/utils.cu

TARGET = milstone2.o

cpu:
	$(CC) $(CFLAGS) -o $(TARGET) $(CPU) $(LDFLAGS)

blas:
	$(CC) $(CFLAGS) -lopenblas "-L/opt/homebrew/opt/openblas/lib" "-I/opt/homebrew/opt/openblas/include" -o $(TARGET) $(BLAS) $(LDFLAGS)

cuda:
	nvcc -arch=sm_86 -Xptxas -O3,-v -o $(TARGET) $(GPU) $(LDFLAGS)

test-cpu:
	./$(TARGET) $(nl) $(nh) $(ne) $(nb) $(alpha)

test-cuda:
	./$(TARGET) $(nl) $(nh) $(ne) $(nb) $(alpha) $(nthreads)

clean:
	rm -f *.o