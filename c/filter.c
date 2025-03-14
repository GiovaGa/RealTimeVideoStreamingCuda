#include <stdio.h>
#include <stdlib.h>

int box_blur(const unsigned char*src, unsigned char *dest, const int height, const int width){
    const int dimx = 10, dimy = dimx;

    for(int ch = 0; ch < 3;++ch){
        for(int i = 0;i < width;++i){
            for(int j = 0;j < height;++j){
                int cnt = 0, tmp = 0;
                for(int di = -(dimx)/2;di <= (dimx+1)/2;++di){
                    if(i+di < 0 || i+di >= width) continue;
                    for(int dj = -(dimy)/2;dj <= (dimy+1)/2;++dj){
                        if(j+dj < 0 || j+dj >= height) continue;
                        if(3*(height*(i+di)+(j+dj))+ch < 0 || 3*(height*(i+di)+(j+dj))+ch >= 3*width*height){
                            fprintf(stderr,"!!! %d,%d,%d -> %d\n",i,j,ch, 3*(width*(i+di)+(j+dj))+ch);
                            exit(-1);
                        }
                        tmp += src[3*(height*(i+di)+(j+dj))+ch];
                        ++cnt;
                    }
                }
                dest[3*(height*(i)+(j))+ch] = tmp/cnt;
            }
        }
    }
    return 0;
}
