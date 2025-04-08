#ifdef DEBUG
#include <stdio.h>
#endif
#ifndef __NVCC__
#include <stdlib.h>
#include <assert.h>
#endif
#include <stdint.h>
#include "filter.h"

static const size_t dimx = 12, dimy = dimx; 

#define minInt(a,b) (((a)<(b))?(a):(b))
#define maxInt(a,b) (((a)<(b))?(b):(a))


#ifdef __NVCC__
__global__ void cuda_box_blur(const uint8_t __restrict__ *src, uint8_t __restrict__ *dest, const size_t width, const size_t height){
    // const int index = threadIdx.x; const int stride = blockDim.x;
    size_t i = blockIdx.x*blockDim.x + threadIdx.x;
    if(i > height) return;
#else
void box_blur(const uint8_t *restrict src, uint8_t *restrict dest, const size_t width, const size_t height){
    for(size_t i = 0;i < height;++i){
#endif
        uint32_t tmp[3] = {0,0,0};

        for(size_t i1 = maxInt(0,i-(dimy)/2);i1 < minInt(i+(dimy+1)/2,height);++i1){
            for(size_t j1 = 0;j1 < minInt((dimx-1)/2,width);++j1){
                for(size_t ch = 0; ch < 3;++ch){
                    tmp[ch] += src[3*(width*i1+j1)+ch];
                }
            }
        }
        const size_t i1 = i-((dimy-1)/2),
                     i2 = i+((dimy+1)/2);
        for(size_t j = 0;j < width;++j){
            const size_t j1 = j-(dimx+1)/2,
                         j2 = j+(dimx-1)/2;

            for(size_t ii = maxInt(0,i1);ii < minInt(i2,height);++ii){
                for(size_t ch = 0; ch < 3;++ch){
                    tmp[ch] += (j2 < width)?src[3*(width*i1+j2)+ch]:0;
                    tmp[ch] -= (j1 < width)?src[3*(width*i1+j1)+ch]:0;
                }
            }

            const uint32_t cnt = (((i2 < height)?i2:height) - ((i1 < height)?i1:0))*
                                 (((j2 < width )?j2:width ) - ((j1 < width )?j1:0));
            for(size_t ch = 0; ch < 3;++ch){
                dest[3*(width*i+j)+ch] = (uint8_t)(tmp[ch]/cnt);
            }
        }
#ifndef __NVCC__
    } // for
#endif
}

#ifdef __NVCC__
const size_t NTHREADS = 512;
extern "C" {
void box_blur(const uint8_t * src, uint8_t * dest, const size_t width, const size_t height){
	cuda_box_blur<<<(height+NTHREADS-1)/NTHREADS,NTHREADS>>>(src,  dest, width, height);
}
}
#endif
