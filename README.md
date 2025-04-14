# Real Time Video Streaming on Linux with Cuda

The aim of this project is to get, process and stream in real time video from a webcam; 
Video4linux, CUDA and FFmpeg will be used for this goal.

This project has been created as a short activity that I did while visiting [Sant’Anna School of Advanced Studies](https://www.santannapisa.it/en), under the supervision of [Prof. Tommaso Cucinotta](https://retis.santannapisa.it/team-members/tommaso-cucinotta/). 

### Dependencies

Necessary libraries are: libv4l, libv4lconvert, libavcodec, libavutil, libavformat, libswresample, libswscale.

### Building
Just run `make` to have a debug version, that prints some information about running time and activates libav logs.

You can have a slightly faster version without printouts by running `make release`.

__Note:__ When building `make` automatically checks if the nvidia compiler `nvcc` is available, if so this compiler is used and provides CUDA acceleration. Otherwise only CPU code is created.
