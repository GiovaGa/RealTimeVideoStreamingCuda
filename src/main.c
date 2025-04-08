#include <assert.h>
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <time.h>

#include <getopt.h>             /* getopt_long() */

#include "capture.h"
#include "utils.h"
#include "filter.h"
#include "streaming.h"


static char            *dev_name;
static int              fd = -1;
static int              out_std = 0; // output images to stdout, 0 to file
static int              force_format = 1;
static int              frame_count = 40;
// static const int 	width = 1920, height = 1080;
static const int 	width = 640, height = 480;
struct v4lconvert_data* data;
static buffer           *buffers = NULL;
static buffer           conv_buffer = {NULL,0}, dest_buffer = {NULL,0}, libav_buffer = {NULL,0};
static struct v4l2_format actual_fmt, wanted_fmt = {
    .type = V4L2_BUF_TYPE_VIDEO_CAPTURE,
    .fmt.pix.width       = width,
    .fmt.pix.height      = height,
    .fmt.pix.pixelformat = V4L2_PIX_FMT_RGB24,
    .fmt.pix.field       = V4L2_FIELD_INTERLACED
};

#ifdef __NVCC__
#include <nppdefs.h>

static buffer d_p = {NULL, 0};

static void process_image(const void *p, int size)
{
	if(d_p.length < size){
		d_p.length = size;
		cudaError_t cerr = cudaMalloc(&d_p.start, size);
		assert(cerr == cudaSuccess);
	}
	if(conv_buffer.length < width*height*3){
		cudaFree(conv_buffer.start);
		cudaFree(dest_buffer.start);
		free(libav_buffer.start);
        conv_buffer.length = dest_buffer.length = width*height*3;
		libav_buffer.length = width*height*2;
		assert(cudaMalloc(&conv_buffer.start, conv_buffer.length) == cudaSuccess);
		assert(cudaMalloc(&dest_buffer.start, dest_buffer.length) == cudaSuccess);
		assert((libav_buffer.start = malloc(libav_buffer.length)) != NULL);
	}
	cudaMemcpy(d_p.start,p,size,cudaMemcpyHostToDevice);

	assert(actual_fmt.fmt.pix.width == width);
	assert(actual_fmt.fmt.pix.height == height);

	const NppiSize nppSize = {.width = width, .height = height};
	assert(NPP_SUCCESS == nppiYUV422ToRGB_8u_C2C3R(
                d_p.start,width*2,
                conv_buffer.start,width*3, nppSize));
	
	box_blur((uint8_t*)conv_buffer.start, (uint8_t*) dest_buffer.start, width, height);
    assert(NPP_SUCCESS == nppiRGBToYUV420_8u_P3R(
                dest_buffer.start,width*3,
                d_p.start,width*2, nppSize));
	cudaMemcpy(libav_buffer.start, dest_buffer.start,libav_buffer.length,cudaMemcpyDeviceToHost);

    send_frame(libav_buffer.start, width, height, YUV420);
}
#else
#ifndef __GNUC__
#warning "Unsupported compiler"
#endif
#include <libv4lconvert.h>

static void process_image(const void *p, int size)
{
    assert(actual_fmt.fmt.pix.width == width);
    assert(actual_fmt.fmt.pix.height == height);
    if(conv_buffer.length < width*height*3){
        free(conv_buffer.start);
        free(dest_buffer.start);
		free(libav_buffer.start);
        conv_buffer.length = dest_buffer.length = width*height*3;
        assert((conv_buffer.start = malloc(conv_buffer.length)) != NULL);
        assert((dest_buffer.start = malloc(dest_buffer.length)) != NULL);
    }

    int ret = v4lconvert_convert(data,&actual_fmt, &wanted_fmt,
        (uint8_t *)p, 2*width*height,
        (uint8_t *)conv_buffer.start,conv_buffer.length);
    if(ret == -1){
        fprintf(stderr,"V4lconvert: %s\n",v4lconvert_get_error_message(data));
        exit(-1);
    }else{
        // conv_buffer.length = r;  ?? 
    }

    box_blur((uint8_t *)conv_buffer.start, (uint8_t *) dest_buffer.start, width, height);

    if(ret == -1){
        fprintf(stderr,"V4lconvert: %s\n",v4lconvert_get_error_message(data));
        exit(-1);
    }
    send_frame(dest_buffer.start, width, height, RGB24);
}
#endif

static int read_frame(const int fd)
{
        struct v4l2_buffer buf;
        unsigned int i;

        CLEAR(buf);

        buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        buf.memory = V4L2_MEMORY_MMAP;

        if (-1 == xioctl(fd, VIDIOC_DQBUF, &buf)) {
                switch (errno) {
                case EAGAIN:
                        return 0;

                case EIO:
                        /* Could ignore EIO, see spec. */
                        /* fall through */

                default:
                        errno_exit("VIDIOC_DQBUF");
                }
        }

        // assert(buf.index < n_buffers);

        process_image(buffers[buf.index].start, buf.bytesused);

        if (-1 == xioctl(fd, VIDIOC_QBUF, &buf))
                errno_exit("VIDIOC_QBUF");
        return 1;
}

static void mainloop(unsigned int count)
{
        int t0, t1;
        while (count-- > 0) {
                t0 = clock();
                for (;;) {
                        fd_set fds;
                        struct timeval tv;
                        int r;

                        FD_ZERO(&fds);
                        FD_SET(fd, &fds);

                        /* Timeout. */
                        tv.tv_sec = 2;
                        tv.tv_usec = 0;

                        r = select(fd + 1, &fds, NULL, NULL, &tv);

                        if (-1 == r) {
                                if (EINTR == errno)
                                        continue;
                                errno_exit("select");
                        }

                        if (0 == r) {
                                fprintf(stderr, "select timeout\n");
                                exit(EXIT_FAILURE);
                        }
                        if (read_frame(fd))
                                break;
                        /* EAGAIN - continue select loop. */
                }
                t1 = clock();
                fprintf(stderr,"Frametime %.3f ms -> %.2f\n",((float)t1-t0)*1000/CLOCKS_PER_SEC, CLOCKS_PER_SEC/((float)t1-t0));
        }
}



static void usage(FILE *fp, int argc, char **argv)
{
        fprintf(fp,
                 "Usage: %s [options]\n\n"
                 "Version 1.3\n"
                 "Options:\n"
                 "-d | --device name   Video device name [%s]\n"
                 "-h | --help          Print this message\n"
                 "-o | --output        Outputs stream to stdout (default to file out)\n"
                 "-f | --format        Do not force format to  %d by %d\n"
                 "-c | --count         Number of frames to grab [%i]\n"
                 "",
                 argv[0], dev_name, width, height, frame_count);
}

static const char short_options[] = "d:hmruofc:";

static const struct option
long_options[] = {
        { "device", required_argument, NULL, 'd' },
        { "help",   no_argument,       NULL, 'h' },
        { "output", no_argument,       NULL, 'o' },
        { "format", no_argument,       NULL, 'f' },
        { "count",  required_argument, NULL, 'c' },
        { 0, 0, 0, 0 }
};

int main(int argc, char **argv)
{
        dev_name = "/dev/video0";

        for (;;) {
                int idx;
                int c;

                c = getopt_long(argc, argv,
                                short_options, long_options, &idx);

                if (-1 == c)
                        break;

                switch (c) {
                case 0: /* getopt_long() flag */
                        break;

                case 'd':
                        dev_name = optarg;
                        break;

                case 'h':
                        usage(stdout, argc, argv);
                        exit(EXIT_SUCCESS);

                case 'o':
                        out_std=1;
                        break;

                case 'f':
                        force_format--;
                        break;

                case 'c':
                        errno = 0;
                        frame_count = strtol(optarg, NULL, 0);
                        if (errno)
                                errno_exit(optarg);
                        break;

                default:
                        usage(stderr, argc, argv);
                        exit(EXIT_FAILURE);
                }
        }

    fd = open_device(dev_name);
    buffers = init_device(fd, force_format,&wanted_fmt,&actual_fmt);
    init_libav(width,height,4); // 4 is buffer size
    start_capturing(fd);
    data = v4lconvert_create(fd);

    mainloop(frame_count);

    v4lconvert_destroy(data);
    stop_capturing(fd);
    uninit_libav();
    uninit_device(buffers);
    close_device(fd); fd = -1;

#ifdef __NVCC__
    cudaFree(conv_buffer.start);
    cudaFree(dest_buffer.start);
#else
    free(conv_buffer.start);
    free(dest_buffer.start);
#endif
    free(libav_buffer.start);


    fprintf(stderr, "\n");
    return 0;
}
