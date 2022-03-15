#ifndef __MALLOC_UTIL_H__
#define __MALLOC_UTIL_H__


#include <stddef.h>


#define MyMalloc(type)   type    *malloc_ ## type ## 1d(size_t n);                      \
                         type   **malloc_ ## type ## 2d(size_t w, size_t h);            \
                         type  ***malloc_ ## type ## 3d(size_t w, size_t h, size_t d);  \
                         void free_ ## type ## 1d(type    *buf);                        \
                         void free_ ## type ## 2d(type   **buf);                        \
                         void free_ ## type ## 3d(type  ***buf);

MyMalloc(int)
MyMalloc(float)
MyMalloc(double)
#undef MyMalloc


#endif

