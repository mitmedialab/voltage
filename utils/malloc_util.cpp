#include <stdio.h>
#include <stdlib.h>
#include <malloc.h>


#define MyMalloc(type)                                                       \
                                                                             \
                                                                             \
type    *malloc_ ## type ## 1d(size_t n)                                     \
{                                                                            \
    type *buf;                                                               \
                                                                             \
    if((buf = (type *)malloc(sizeof(type) * n)) == NULL)                     \
    {                                                                        \
        fprintf(stderr, "failed to allocate memory\n"); exit(1);             \
    }                                                                        \
    return buf;                                                              \
}                                                                            \
                                                                             \
type   **malloc_ ## type ## 2d(size_t w, size_t h)                           \
{                                                                            \
    type **buf;                                                              \
                                                                             \
    if((buf = (type **)malloc(sizeof(type *) * w)) == NULL)                  \
    {                                                                        \
        fprintf(stderr, "failed to allocate memory\n"); exit(1);             \
    }                                                                        \
    if((buf[0] = (type *)malloc(sizeof(type) * w * h)) == NULL)              \
    {                                                                        \
        fprintf(stderr, "failed to allocate memory\n"); exit(1);             \
    }                                                                        \
    for(size_t i = 1; i < w; i++) buf[i] = buf[0] + i * h;                   \
    return buf;                                                              \
}                                                                            \
                                                                             \
type  ***malloc_ ## type ## 3d(size_t w, size_t h, size_t d)                 \
{                                                                            \
    type ***buf;                                                             \
                                                                             \
    if((buf = (type ***)malloc(sizeof(type **) * w)) == NULL)                \
    {                                                                        \
        fprintf(stderr, "failed to allocate memory\n"); exit(1);             \
    }                                                                        \
    if((buf[0] = (type **)malloc(sizeof(type *) * w * h)) == NULL)           \
    {                                                                        \
        fprintf(stderr, "failed to allocate memory\n"); exit(1);             \
    }                                                                        \
    for(size_t i = 1; i < w; i++) buf[i] = buf[0] + i * h;                   \
    if((buf[0][0] = (type *)malloc(sizeof(type) * w * h * d)) == NULL)       \
    {                                                                        \
        fprintf(stderr, "failed to allocate memory\n"); exit(1);             \
    }                                                                        \
    for(size_t i = 1; i < w * h; i++) buf[0][i] = buf[0][0] + i * d;         \
    return buf;                                                              \
}                                                                            \
                                                                             \
                                                                             \
void free_ ## type ## 1d(type    *buf)                                       \
{                                                                            \
    free(buf);                                                               \
}                                                                            \
                                                                             \
void free_ ## type ## 2d(type   **buf)                                       \
{                                                                            \
    free(buf[0]);                                                            \
    free(buf);                                                               \
}                                                                            \
                                                                             \
void free_ ## type ## 3d(type  ***buf)                                       \
{                                                                            \
    free(buf[0][0]);                                                         \
    free(buf[0]);                                                            \
    free(buf);                                                               \
}


MyMalloc(int)
MyMalloc(float)
MyMalloc(double)
#undef MyMalloc

