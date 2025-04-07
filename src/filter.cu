#ifdef DEBUG
#include <stdio.h>
#endif
#ifndef __NVCC__
#include <stdlib.h>
#include <assert.h>
#include <omp.h>
#endif
#include <stdint.h>
#include "filter.h"

static const int dimx = 12, dimy = dimx; 

#define minInt(a,b) (((a)<(b))?(a):(b))
#define maxInt(a,b) (((a)<(b))?(b):(a))


#ifdef __NVCC__
__global__
void cuda_box_blur(const uint8_t __restrict__ *src, uint8_t __restrict__ *dest, const int height, const int width){
    // const int index = threadIdx.x; const int stride = blockDim.x;
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if(i > width) return;
#else
void box_blur(const uint8_t *restrict src, uint8_t *restrict dest, const int height, const int width){
    // #pragma omp parallel for
    for(int i = 0;i < width;++i){
#endif
        int32_t tmp[3] = {0,0,0};

        for(int i1 = maxInt(0,i-(dimx)/2);i1 < minInt(i+(dimx+1)/2,width);++i1){
            for(int j1 = 0;j1 < minInt((dimy-1)/2,height);++j1){
                for(int ch = 0; ch < 3;++ch){
                    tmp[ch] += src[3*(height*i1+j1)+ch];
                }
            }
        }
        for(int j = 0;j < height;++j){
            
            const int j1 = j-(dimy+1)/2,
                      j2 = j+(dimy-1)/2;
            for(int i1 = maxInt(0,i-(dimx-1)/2);i1 < minInt(i+(dimx+1)/2,width);++i1){
                for(int ch = 0; ch < 3;++ch){
                    tmp[ch] -= (j1 >= 0    )?src[3*(height*i1+j1)+ch]:0;
                    tmp[ch] += (j2 < height)?src[3*(height*i1+j2)+ch]:0;
                }
            }
            const int cnt = (minInt(i+((dimx+1)/2),width)-maxInt(0,i-((dimx-1)/2)))*(minInt(j+((dimy+1)/2),height)-maxInt(0,j-(dimy-1)/2));

            for(int ch = 0; ch < 3;++ch){
                dest[3*(height*i+j)+ch] = (uint8_t)(tmp[ch]/cnt);
            }
        }
#ifndef __NVCC__
    } // for
#endif
}

#ifdef __NVCC__
const int NTHREADS = 512;
extern "C" {
void box_blur(const uint8_t * src, uint8_t * dest, const int height, const int width){
	cuda_box_blur<<<(width+NTHREADS-1)/NTHREADS,NTHREADS>>>(src,  dest, height, width);
}
}
#endif
