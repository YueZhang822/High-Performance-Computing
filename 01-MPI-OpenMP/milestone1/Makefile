CC = clang

CFLAGS = -Wall -O3 -flto -march=native -mtune=native -lm
TARGET = advection

# test input parameter setting
N = 400
NT = 20000
L = 1.0
T = 1.0e6
u = 5.0e-7
v = 2.85e-7

.PHONY: all clean test

all: $(TARGET)

$(TARGET): $(TARGET).c 
		$(CC) $(CFLAGS) -o $(TARGET).o $(TARGET).c 
		mkdir -p results/data
		mkdir -p results/images

clean:
		rm -rf $(TARGET).o
		rm -rf results/data

test:
		./$(TARGET).o $(N) $(NT) $(L) $(T) $(u) $(v)