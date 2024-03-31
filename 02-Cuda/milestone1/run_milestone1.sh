PROG=milestone1.o

make clean
make all
make test

python3 visualize.py

rm $PROG