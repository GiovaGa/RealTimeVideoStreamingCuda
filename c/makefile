CC=gcc
CLIBS=-lv4l2 -lv4lconvert
CFLAGS=-O0 -fsanitize=address -fsanitize=undefined -g -Wall $(CLIBS)

all: capture main

capture: capture.c 
	$(CC) $(CFLAGS) -o capture.o capture.c

main: main.c 
	$(CC) $(CFLAGS) -o main.o main.c

clean:
	rm *.o
