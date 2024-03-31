make clean

make serial-sp
make test

make serial-dp
make test

make openmp-sp
make test

make openmp-dp
make test

make gpu-sp
make test

make gpu-dp
make test

python3 visualize.py