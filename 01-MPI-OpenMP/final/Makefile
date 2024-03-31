CC = mpicc

CFLAGS = -O3 -flto -march=native -mtune=native -lm
TARGET = advection

# test input parameter setting
N = 400
L = 1.0
T = 1.0
delta_t = 1.25e-4
m = 2
n = 2
num_threads = 3

.PHONY: all clean test

all: $(TARGET)

$(TARGET): $(TARGET).c 
		mkdir -p results/data
		mkdir -p results/images
		$(CC) $(CFLAGS) -fopenmp -o $(TARGET).o $(TARGET).c 

clean:
		rm -rf $(TARGET).o
		rm -rf results

test:
		mpirun -np 4 ./$(TARGET).o $(N) $(L) $(T) $(delta_t) $(m) $(n) $(num_threads)