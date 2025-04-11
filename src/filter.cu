#ifdef DEBUG
#include <stdio.h>
#endif
#ifndef __NVCC__
#include <stdlib.h>
#include <assert.h>
#endif
#include <stdint.h>
#include "filter.h"

static const size_t dimx = 11, dimy = dimx; 

#define minInt(a,b) (((a)<(b))?(a):(b))
#define maxInt(a,b) (((a)<(b))?(b):(a))

#define index(i,j,ch) (3*(width*(i)+(j))+(ch))

#ifdef __NVCC__
__global__ void cuda_box_blur(const uint8_t __restrict__ *src, uint8_t __restrict__ *dest, const size_t width, const size_t height){
    const int index = threadIdx.x; const int stride = blockDim.x;
    // const int i = blockIdx.x * blockDim.x + threadIdx.x;
    for(size_t i =  threadIdx.x;i < height;i += blockDim.x){
#else
void box_blur(const uint8_t *restrict src, uint8_t *restrict dest, const size_t width, const size_t height){
    for(size_t i = 0;i < height;++i){
#endif
        const size_t i1 = maxInt(i-((int)(dimy)/2),0),
                     i2 = minInt(i+((dimy+1)/2),height);
        size_t j1 =  maxInt(0-((int)(dimx)/2),0), j2 = minInt(0+((dimx+1)/2),width);
        int32_t tmp[3] = {0,0,0};
        // int32_t cnt = 0;

        for(int jj = j1;jj < j2;++jj){
            for(size_t ii = i1;ii < i2;++ii){
                for(size_t ch = 0; ch < 3;++ch){
                    tmp[ch] += src[index(ii,jj,ch)];
                }
                // ++cnt;
            }
        }
        for(int32_t j = 0;j < width;++j){
            // assert(cnt > 0);
            // assert(cnt == (i2-i1)*(j2-j1));

            const int32_t cnt = (i2-i1)*(j2-j1);

#ifdef DEBUG
            // fprintf(stderr,"%d,%d -> ",i,j); fprintf(stderr,"%d == (%d-%d)*(%d-%d)\n",cnt,i2,i1,j2,j1);
            assert(tmp[0] >= 0);
            assert(tmp[0] <= 256*cnt);
#endif
            for(size_t ch = 0; ch < 3;++ch){
                dest[index(i,j,ch)] = (uint8_t)((tmp[ch])/cnt);
            }
            if(j2 < width){
                for(size_t ii = i1;ii < i2;++ii){
                    for(size_t ch = 0; ch < 3;++ch){
                        tmp[ch] += src[index(ii,j2,ch)];
                    } // ++cnt;
                }
                ++j2;
            }
            if(j2-j1 >= dimx){
                for(size_t ii = i1;ii < i2;++ii){
                    for(size_t ch = 0; ch < 3;++ch){
                        tmp[ch] -= src[index(ii,j1,ch)];
                    } // --cnt;
                }
                ++j1;
            }
        }
#ifndef __NVCC__
    } // for
#endif
}

#ifdef __NVCC__
__global__ void cuda2d_box_blur(const uint8_t __restrict__ *src, uint8_t __restrict__ *dest, const size_t width, const size_t height){
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int j = blockIdx.y * blockDim.y + threadIdx.y;
    if(i > height || j > width) return;
    const size_t i1 = maxInt(i-((dimy)/2),0), i2 = minInt(i+((dimy+1)/2),height);
    const size_t j1 = maxInt(j-((dimx)/2),0), j2 = minInt(j+((dimx+1)/2),width);
    int32_t tmp[3] = {0,0,0};

    for(int jj = j1;jj < j2;++jj){
        for(size_t ii = i1;ii < i2;++ii){
            for(size_t ch = 0; ch < 3;++ch){
                tmp[ch] += src[index(ii,jj,ch)];
            }
        }
    }
    const int32_t cnt = (i2-i1)*(j2-j1);

    for(size_t ch = 0; ch < 3;++ch){
        dest[index(i,j,ch)] = (uint8_t)((tmp[ch])/cnt);
    }
}



const size_t NTHREADS = 1; // provare con pochissime
extern "C" {
void box_blur(const uint8_t * src, uint8_t * dest, const size_t width, const size_t height){
	cuda_box_blur<<<(height+NTHREADS-1)/NTHREADS,NTHREADS>>>(src,  dest, width, height);
    return;

    const dim3 threadsPerBlock(16, 16);
    const dim3 numBlocks(height / threadsPerBlock.x, width / threadsPerBlock.y);
	cuda2d_box_blur<<<numBlocks,threadsPerBlock>>>(src,  dest, width, height);
}
}
#endif
