RANDOM = random

# benchmark problem setting
n_ray = 1000000000
n_grid = 1000
n_block = 128
ntpb = 128
nthreads = 16

serial-sp: serial-sp.c
	gcc -Wall -O3 -flto -march=native -mtune=native -lm -o ray.o serial-sp.c $(RANDOM).c
	mkdir -p results/serial-sp

serial-dp: serial-dp.c
	gcc -Wall -O3 -flto -march=native -mtune=native -lm -o ray.o serial-dp.c $(RANDOM).c
	mkdir -p results/serial-dp

opnemp-sp: openmp-sp.c
	gcc -Wall -O3 -flto -fopenmp -march=native -mtune=native -lm -o ray.o openmp-sp.c $(RANDOM).c
	mkdir -p results/openmp-dp

openmp-dp: openmp-dp.c
	gcc -Wall -O3 -flto -fopenmp -march=native -mtune=native -lm -o ray.o openmp-dp.c $(RANDOM).c
	mkdir -p results/openmp-dp

gpu-sp: gpu-sp.cu
	nvcc -ccbin=gcc -Xptxas -O3,-v -o ray.o gpu-sp.cu
	mkdir -p results/gpu-sp

gpu-dp: gpu-dp.cu
	nvcc -ccbin=gcc -Xptxas -O3,-v -o ray.o gpu-dp.cu
	mkdir -p results/gpu-dp

clean:
	rm -f *.o
	rm -rf results

test:
	./ray.o $(n_ray) $(n_grid) $(n_block) $(ntpb) $(nthreads)