#ifndef CAPTURE_H
#define CAPTURE_H

#include <linux/videodev2.h>
#include <libv4l2.h>
#include <libv4lconvert.h>

typedef struct bu{
        void   *start;
        size_t  length;
} buffer;

void start_capturing(const int);
void stop_capturing(const int);
int open_device(const char*);
void close_device(const int);
void*init_device(const int fd, const int,struct v4l2_format *, struct v4l2_format *);
void uninit_device(buffer *buffers);

#endif
