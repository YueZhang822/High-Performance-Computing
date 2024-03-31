CC = gcc

CFLAGS = -Wall -O3 -flto -march=native -mtune=native
LDFLAGS = -lm

nl = 2
nh = 10
ne = 100
nb = 128
alpha = 0.1

CPU = milestone1.c utils.c

TARGET = milstone1.o

cpu:
	$(CC) $(CFLAGS) -o $(TARGET) $(CPU) $(LDFLAGS)

test:
	./$(TARGET) $(nl) $(nh) $(ne) $(nb) $(alpha)

clean:
	rm -f *.o
