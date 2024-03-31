PROG=milestone2.o

make clean
make all
make test

python3 ../utils/visualize.py

rm $PROG