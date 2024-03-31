PROG=advection.o

make clean
make all
make test

./$PROG "$@"
python3 plot.py

rm $PROG