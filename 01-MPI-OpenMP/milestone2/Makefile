CC = gcc

CFLAGS = -Wall -O3 -flto -march=native -mtune=native -lm
TARGET = advection

# test input parameter setting
N = 400
L = 1.0
T = 1.0
u = 1.0
v = 1.0
num_threads = 4

.PHONY: all clean test

all: $(TARGET)

lax-serial: $(TARGET).c
	$(CC) $(CFLAGS) -fopenmp -DSERIAL -DLAX -o $(TARGET).o $(TARGET).c
	mkdir -p results/data

lax-parallel: $(TARGET).c
	$(CC) $(CFLAGS) -fopenmp -DLAX -o $(TARGET).o $(TARGET).c
	mkdir -p results/data

first-serial: $(TARGET).c
	$(CC) $(CFLAGS) -fopenmp -DSERIAL -DFIRST -o $(TARGET).o $(TARGET).c
	mkdir -p results/data

first-parallel: $(TARGET).c
	$(CC) $(CFLAGS) -fopenmp -DFIRST -o $(TARGET).o $(TARGET).c
	mkdir -p results/data

second-serial: $(TARGET).c
	$(CC) $(CFLAGS) -fopenmp -DSERIAL -DSECOND -o $(TARGET).o $(TARGET).c
	mkdir -p results/data

second-parallel: $(TARGET).c
	$(CC) $(CFLAGS) -fopenmp -DSECOND -o $(TARGET).o $(TARGET).c
	mkdir -p results/data

clean:
	rm -rf $(TARGET).o
	rm -rf results/data
	rm -rf results/images

test:
	./$(TARGET).o $(N) $(L) $(T) $(u) $(v) $(num_threads)
