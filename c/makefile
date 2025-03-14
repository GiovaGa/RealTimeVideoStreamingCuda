CC=gcc
CFLAGS=-O2 -g -Wall


all: capture0 capture main

capture0: capture0.c 
	$(CC) $(CLFAGS) -o capture0.o capture0.c

capture: capture.c 
	$(CC) $(CLFAGS) -o capture.o capture.c

main: main.c 
	$(CC) $(CLFAGS) -o main.o main.c

clean:
	rm a.out
