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
    const int dimx = 12, dimy = dimx; 

    // #pragma omp parallel for
    for(int i = 0;i < width;++i){
        int tmp[3] = {0,0,0};

        for(int i1 = max(0,i-(dimx)/2);i1 < min(i+(dimx+1)/2,width);++i1){
            for(int j1 = 0;j1 < min((dimy-1)/2,height);++j1){
                #pragma omp simd
                for(int ch = 0; ch < 3;++ch){
                    tmp[ch] += src[3*(height*i1+j1)+ch];
                }
            }
        }
        for(int j = 0;j < height;++j){
            
            const int j1 = j-(dimy+1)/2,
                      j2 = j+(dimy-1)/2;
            for(int i1 = max(0,i-(dimx-1)/2);i1 < min(i+(dimx+1)/2,width);++i1){
                #pragma omp simd
                for(int ch = 0; ch < 3;++ch){
                    tmp[ch] -= (j1 >= 0)?   src[3*(height*i1+j1)+ch]:0;
                    tmp[ch] += (j2 < height)?src[3*(height*i1+j2)+ch]:0;
                }
            }
            const int cnt = (min(i+((dimx+1)/2),width)-max(0,i-((dimx-1)/2)))*(min(j+((dimy+1)/2),height)-max(0,j-(dimy-1)/2));


            // assert(cnt > 0);
            // assert(tmp[0] > 0 && tmp[1] > 0 && tmp[2] > 0);
            // fprintf(stderr,"%d,%d | %d -- %d\t",i,j,j1,j2);
            // fprintf(stderr,"%d,%d,%d\n",src[3*(height*i+j)+0],src[3*(height*i+j)+1],src[3*(height*i+j)+2]);
            // fprintf(stderr,"%d,%d,%d %d\t",tmp[0],tmp[1],tmp[2],cnt); fprintf(stderr,"->\t%d,%d,%d\n",(unsigned char)tmp[0]/cnt,(unsigned char)tmp[1]/cnt,(unsigned char)tmp[2]/cnt);

            #pragma omp simd
            for(int ch = 0; ch < 3;++ch){
                dest[3*(height*i+j)+ch] = (unsigned char)(((float)tmp[ch])/cnt);
            }
        }
    }
    return 0;
}
