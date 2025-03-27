#ifndef UTILS_H
#define UTILS_H
#define CLEAR(x) memset(&(x), 0, sizeof(x))

void errno_exit(const char *s);
void av_errno_exit(const char *s, const int ret);
int xioctl(int fh, int request, void *arg);

#endif
