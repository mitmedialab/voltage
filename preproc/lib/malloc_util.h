#ifndef __MALLOC_UTIL_H__
#define __MALLOC_UTIL_H__


#define MyMalloc(type)   type    *malloc_ ## type ## 1d(int n);                       \
                         type   **malloc_ ## type ## 2d(int w, int h);                \
                         type  ***malloc_ ## type ## 3d(int w, int h, int d);         \
                         type ****malloc_ ## type ## 4d(int w, int h, int d, int t);  \
                         void free_ ## type ## 1d(type    *buf);                      \
                         void free_ ## type ## 2d(type   **buf);                      \
                         void free_ ## type ## 3d(type  ***buf);                      \
                         void free_ ## type ## 4d(type ****buf);

MyMalloc(int)
MyMalloc(float)
MyMalloc(double)
#undef MyMalloc


#endif

