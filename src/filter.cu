#ifdef DEBUG
#include <stdio.h>
#endif
#ifndef __NVCC__
#include <stdlib.h>
#include <assert.h>
#endif
#include <stdint.h>
#include "filter.h"

static const int32_t dimx = 11, dimy = dimx; 

#define minInt(a,b) (((a)<(b))?(a):(b))
#define maxInt(a,b) (((a)<(b))?(b):(a))

#define index(i,j,ch) (3*(width*(i)+(j))+(ch))

#ifdef __NVCC__
__global__ void cuda_box_blur(const uint8_t __restrict__ *src, uint8_t __restrict__ *dest, const size_t width, const size_t height){
    // const int index = threadIdx.x; const int stride = blockDim.x;
    size_t i = blockIdx.x*blockDim.x + threadIdx.x;
    if(i > height) return;
#else
void box_blur(const uint8_t *restrict src, uint8_t *restrict dest, const size_t width, const size_t height){
    for(int32_t i = 0;i < height;++i){
#endif
        const int32_t i1 = maxInt(i-((dimy)/2),0),
                      i2 = minInt(i+((dimy+1)/2),height);
        int32_t cnt = 0;
        int32_t tmp[3] = {0,0,0};
        int32_t j1 = 0,
                j2 = ((dimx-1)/2);

        for(size_t jj = j1;jj < j2;++jj){
            for(size_t ii = i1;ii < i2;++ii){
                for(size_t ch = 0; ch < 3;++ch){
                    tmp[ch] += src[index(ii,jj,ch)];
                }
                ++cnt;
            }
        }

        for(int32_t j = 0;j < width;++j){
            j1 = j-((dimx)/2), j2 = j+((dimx+1)/2);

            for(size_t ii = i1;ii < i2;++ii){
                if(j2 < width){
                    for(size_t ch = 0; ch < 3;++ch){
                        tmp[ch] += src[index(ii,j2,ch)];
                    }
                    ++cnt;
                }
                if(j1 > 0){
                    for(size_t ch = 0; ch < 3;++ch){
                        tmp[ch] -= src[index(ii,j1-1,ch)];
                    }
                    --cnt;
                }
            }

            const int32_t jj1 = maxInt(j1,0),
                          jj2 = minInt(j2,width);

            assert(cnt > 0);
            fprintf(stderr,"%d == (%d-%d)*(%d-%d)\n",cnt,i2,i1,jj2,jj1);
            assert(cnt == (i2-i1)*(jj2-jj1));

            assert(tmp[0] >= 0);
            assert(tmp[0] <= 256*cnt);
            cnt = (i2-i1)*(jj2-jj1);

            for(size_t ch = 0; ch < 3;++ch){
                dest[index(i,j,ch)] = (uint8_t)((tmp[ch])/cnt);
            }
        }
#ifndef __NVCC__
    } // for
#endif
}

#ifdef __NVCC__
const size_t NTHREADS = 1; // provare con pochissim
extern "C" {
void box_blur(const uint8_t * src, uint8_t * dest, const size_t width, const size_t height){
	cuda_box_blur<<<(height+NTHREADS-1)/NTHREADS,NTHREADS>>>(src,  dest, width, height);
}
}
#endif
