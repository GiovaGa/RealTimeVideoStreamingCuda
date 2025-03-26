#ifndef STREAMING_H
#define STREAMING_H


void init_libav(const int width, const int height, const int count);
void uninit_libav();
int send_frame(void*, const int, const int);



#endif
