#ifndef UTILS_H
#define UTILS_H
#define CLEAR(x) memset(&(x), 0, sizeof(x))

void errno_exit(const char *s);
int xioctl(int fh, int request, void *arg);

#endif
