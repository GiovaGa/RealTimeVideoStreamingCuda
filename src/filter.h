#ifndef FILTER_H
#define FILTER_H

#ifdef __CUDACC__
extern "C" {
#endif
void box_blur(const uint8_t *, uint8_t *, const int, const int);
#ifdef __CUDACC__
} // extern "C"
#endif

#endif
