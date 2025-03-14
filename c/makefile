CC=gcc
CLIBS=-lv4l2 -lv4lconvert
CFLAGS=-O0 -fsanitize=address -fsanitize=undefined -g -Wall

all: capture main

capture: filter
	$(CC) $(CFLAGS) $(CLIBS) -o capture.o capture.c

filter:
	$(CC) $(CFLAGS) -o filter.o -c filter.c

main:
	$(CC) $(CFLAGS) $(CLIBS) -o main.o main.c

clean:
	rm *.o
