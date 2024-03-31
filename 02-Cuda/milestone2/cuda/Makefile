CC = nvcc

CFLAGS = -ccbin=gcc -Xptxas -O3,-v 
TARGET = milestone2

# benchmark problem setting
Lx = 4
Ly = 4
Lz = -1
Wy = 2
Wmax = 2
Cx = 0
Cy = 12
Cz = 0
R = 6
n_ray = 1000000000
n_grid = 1000
nthreads = 128
nblocks = 128

.PHONY: all clean test 

all: $(TARGET)

$(TARGET): $(TARGET).cu
			$(CC) $(CFLAGS) -o $(TARGET).o $(TARGET).cu
			mkdir -p results

clean:
			rm -f $(TARGET).o
			rm -rf results

test:
			./$(TARGET).o $(Lx) $(Ly) $(Lz) $(Wy) $(Wmax) $(Cx) $(Cy) $(Cz) $(R) $(n_ray) $(n_grid) $(nthreads) $(nblocks)