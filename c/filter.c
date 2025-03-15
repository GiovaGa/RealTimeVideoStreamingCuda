#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <omp.h>

int min(const int a,const int b){
    return ((a<b)?a:b);
}
int max(const int a,const int b){
    return ((a<b)?b:a);
}


int box_blur(const unsigned char*src, unsigned char *dest, const int height, const int width){
    const int dimx = 10, dimy = dimx;

    // #pragma omp parallel for collapse(1)
    for(int i = 0;i < width;++i){
        for(int j = 0;j < height;++j){
            int tmp[3] = {0,0,0};
            
            for(int i1 = max(0,i-(dimx)/2);i1 < min(i+(dimx+1)/2,width);++i1){
                for(int j1 = max(j-(dimy)/2,0);j1 < min(j+(dimy+1)/2,height);++j1){
                    #pragma omp simd
                    for(int ch = 0; ch < 3;++ch){
                        tmp[ch] += src[3*(height*i1+j1)+ch];
                    }
                }
            }
            const int cnt = (min(i+((dimx+1)/2),width)-max(0,i-(dimx/2)))*(min(j+((dimy+1)/2),height)-max(0,j-(dimy)/2));
            assert(cnt > 0);
            #pragma omp simd
            for(int ch = 0; ch < 3;++ch){
                dest[3*(height*i+j)+ch] = tmp[ch]/cnt;
            }
        }
    }
    return 0;
}
