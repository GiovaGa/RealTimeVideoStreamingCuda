CC=gcc
CLIBS=-lv4l2 -lv4lconvert
CFLAGS=-O0 -fsanitize=address -fsanitize=undefined -g -Wall $(CLIBS)

all: capture0 capture main

capture0: capture0.c 
	$(CC) $(CFLAGS) -o capture0.o capture0.c

capture: capture.c 
	$(CC) $(CFLAGS) -o capture.o capture.c

main: main.c 
	$(CC) $(CFLAGS) -o main.o main.c

clean:
	rm *.o
