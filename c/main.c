#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>

#include <fcntl.h>              /* low-level i/o */
#include <unistd.h>
#include <errno.h>
// #include <sys/stat.h>
// #include <sys/types.h>
// #include <sys/time.h>
// #include <sys/mman.h>
// #include <sys/ioctl.h>

#include <linux/videodev2.h>
#include <libv4l2.h>
#include <libv4lconvert.h>

static char *dev_name = "/dev/video0";
static int fd = -1;

int main(){
    fd = open(dev_name, O_RDWR /* required */ | O_NONBLOCK, 0);
    char* src = "asdfghjk";
    char*dest = malloc(sizeof(char)*strlen(src));
    v4lconvert_yuyv_to_rgb24(src, dest, 1, 1, 0);
}
