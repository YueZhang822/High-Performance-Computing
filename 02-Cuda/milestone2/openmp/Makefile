CC = gcc

CFLAGS = -Wall -O3 -flto -fopenmp -march=native -mtune=native -lm
TARGET = milestone2
RANDOM = random

# test input parameter setting
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
n_thread = 16

.PHONY: all clean test 

all: $(TARGET)

$(TARGET): $(TARGET).c
			$(CC) $(CFLAGS) -o $(TARGET).o $(TARGET).c $(RANDOM).c
			mkdir -p results

clean:
			rm -f $(TARGET).o
			rm -rf results

test:
			./$(TARGET).o $(Lx) $(Ly) $(Lz) $(Wy) $(Wmax) $(Cx) $(Cy) $(Cz) $(R) $(n_ray) $(n_grid) $(n_thread)