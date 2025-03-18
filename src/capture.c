/*
 *  V4L2 video capture example
 *
 *  This program can be used and distributed without restrictions.
 *
 *      This program is provided with the V4L2 API
 * see https://linuxtv.org/docs.php for more information
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>

#include <fcntl.h>              /* low-level i/o */
#include <unistd.h>
#include <errno.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/time.h>
#include <sys/mman.h>

#include <linux/videodev2.h>
#include <libv4l2.h>

#include "utils.h"
#include "capture.h"

static unsigned int n_buffers = 0;

void stop_capturing(const int fd)
{
        enum v4l2_buf_type type;

        type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        if (-1 == xioctl(fd, VIDIOC_STREAMOFF, &type))
                errno_exit("VIDIOC_STREAMOFF");
}

void start_capturing(const int fd)
{
        unsigned int i;
        enum v4l2_buf_type type;

        for (i = 0; i < n_buffers; ++i) {
                struct v4l2_buffer buf;

                CLEAR(buf);
                buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
                buf.memory = V4L2_MEMORY_MMAP;
                buf.index = i;

                if (-1 == xioctl(fd, VIDIOC_QBUF, &buf))
                        errno_exit("VIDIOC_QBUF");
        }
        type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        if (-1 == xioctl(fd, VIDIOC_STREAMON, &type))
                errno_exit("VIDIOC_STREAMON");
}

void uninit_device(buffer *buffers)
{
        unsigned int i;

        for (i = 0; i < n_buffers; ++i)
                if (-1 == v4l2_munmap(buffers[i].start, buffers[i].length))
                        errno_exit("munmap");
        free(buffers);
}


void* init_mmap(const int fd)
{
        struct v4l2_requestbuffers req;

        CLEAR(req);

        req.count = 4;
        req.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        req.memory = V4L2_MEMORY_MMAP;

        if (-1 == xioctl(fd, VIDIOC_REQBUFS, &req)) {
                if (EINVAL == errno) {
                        fprintf(stderr, "Selected device does not support memory mapping\n");
                        exit(EXIT_FAILURE);
                } else {
                        errno_exit("VIDIOC_REQBUFS");
                }
        }

        if (req.count < 2) {
                fprintf(stderr, "Insufficient buffer memory on selected device\n");
                exit(EXIT_FAILURE);
        }

        buffer* buffers = calloc(req.count, sizeof(*buffers));

        if (!buffers) {
                fprintf(stderr, "Out of memory\n");
                exit(EXIT_FAILURE);
        }

        for (n_buffers = 0; n_buffers < req.count; ++n_buffers) {
                struct v4l2_buffer buf;

                CLEAR(buf);

                buf.type        = V4L2_BUF_TYPE_VIDEO_CAPTURE;
                buf.memory      = V4L2_MEMORY_MMAP;
                buf.index       = n_buffers;

                if (-1 == xioctl(fd, VIDIOC_QUERYBUF, &buf))
                        errno_exit("VIDIOC_QUERYBUF");

                buffers[n_buffers].length = buf.length;
                buffers[n_buffers].start =
                        v4l2_mmap(NULL /* start anywhere */,
                              buf.length,
                              PROT_READ | PROT_WRITE /* required */,
                              MAP_SHARED /* recommended */,
                              fd, buf.m.offset);

                if (MAP_FAILED == buffers[n_buffers].start)
                        errno_exit("mmap");
        }
        return buffers;
}

void*init_device(const int fd, const int force_format,struct v4l2_format*wanted_fmt, struct v4l2_format*actual_fmt)
{
        struct v4l2_capability cap;
        struct v4l2_cropcap cropcap;
        struct v4l2_crop crop;

        if (-1 == xioctl(fd, VIDIOC_QUERYCAP, &cap)) {
                if (EINVAL == errno) {
                        fprintf(stderr, "Selected file descriptor is no V4L2 device\n");
                        exit(EXIT_FAILURE);
                } else {
                        errno_exit("VIDIOC_QUERYCAP");
                }
        }

        if (!(cap.capabilities & V4L2_CAP_VIDEO_CAPTURE)) {
                fprintf(stderr, "Selected device is no video capture device\n");
                exit(EXIT_FAILURE);
        }

        if (!(cap.capabilities & V4L2_CAP_STREAMING)) {
                fprintf(stderr, "Selected device does not support streaming i/o\n");
                exit(EXIT_FAILURE);
        }

        /* Select video input, video standard and tune here. */


        CLEAR(cropcap);

        cropcap.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;

        if (0 == xioctl(fd, VIDIOC_CROPCAP, &cropcap)) {
                crop.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
                crop.c = cropcap.defrect; /* reset to default */

                if (-1 == xioctl(fd, VIDIOC_S_CROP, &crop)) {
                        switch (errno) {
                        case EINVAL:
                                /* Cropping not supported. */
                                break;
                        default:
                                /* Errors ignored. */
                                break;
                        }
                }
        } else {
                /* Errors ignored. */
        }

        actual_fmt->type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        if (force_format) {
                actual_fmt->fmt.pix.width       = wanted_fmt->fmt.pix.width;
                actual_fmt->fmt.pix.height      = wanted_fmt->fmt.pix.height;

                if (-1 == xioctl(fd, VIDIOC_S_FMT, actual_fmt))
                    errno_exit("VIDIOC_S_FMT");
                if (actual_fmt->fmt.pix.pixelformat != V4L2_PIX_FMT_RGB24) {
                    // fprintf(stderr,"Libv4l didn't accept RGB24 format.\n");
                }

                /* Note VIDIOC_S_FMT may change width and height. */
        } else {
                /* Preserve original settings as set by v4l2-ctl for example */
                *wanted_fmt = *actual_fmt;
        }
        if (-1 == xioctl(fd, VIDIOC_G_FMT, actual_fmt))
            errno_exit("VIDIOC_G_FMT");

        /* Buggy driver paranoia. */
        /*
        unsigned int min = wanted_fmt->fmt.pix.width * 2;
        if (actual_fmt->fmt.pix.bytesperline < min){
                actual_fmt->fmt.pix.bytesperline = wanted_fmt->fmt.pix.bytesperline = min;
        }
        min = actual_fmt->fmt.pix.bytesperline * actual_fmt->fmt.pix.height;
        if (actual_fmt->fmt.pix.sizeimage < min){
                actual_fmt->fmt.pix.bytesperline = wanted_fmt->fmt.pix.bytesperline = min;
        }
        */

        return init_mmap(fd);
}

void close_device(const int fd)
{
        if (-1 == v4l2_close(fd))
                errno_exit("close");

        return;
}

int open_device(const char* dev_name)
{
        struct stat st;
        int fd;

        if (-1 == stat(dev_name, &st)) {
                fprintf(stderr, "Cannot identify '%s': %d, %s\n",
                         dev_name, errno, strerror(errno));
                exit(EXIT_FAILURE);
        }

        if (!S_ISCHR(st.st_mode)) {
                fprintf(stderr, "%s is no devicen", dev_name);
                exit(EXIT_FAILURE);
        }

        fd = v4l2_open(dev_name, O_RDWR /* required */ | O_NONBLOCK, 0);

        if (-1 == fd) {
                fprintf(stderr, "Cannot open '%s': %d, %s\n",
                         dev_name, errno, strerror(errno));
                exit(EXIT_FAILURE);
        }
        return fd;
}

