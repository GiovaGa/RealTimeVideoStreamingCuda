#include <errno.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <sys/ioctl.h>

#include <libavcodec/avcodec.h> // av_err2str

void errno_exit(const char *s)
{
    fprintf(stderr, "%s error %d, %s\n", s, errno, strerror(errno));
    exit(EXIT_FAILURE);
}

void av_errno_exit(const char *s, const int ret)
{
    fprintf(stderr, "%s error %d, %s\n", s, ret, av_err2str(ret));
    exit(EXIT_FAILURE);
}

int xioctl(int fh, int request, void *arg)
{
    int r;

    do {
            r = ioctl(fh, request, arg);
    } while (-1 == r && EINTR == errno);

    return r;
}
