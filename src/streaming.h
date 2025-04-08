#ifndef STREAMING_H
#define STREAMING_H

enum format_enum{
    RGB24, YUV420
};

void init_libav(const int width, const int height, const int count);
void uninit_libav();
int send_frame(void*, const int, const int, const enum format_enum);

#endif
