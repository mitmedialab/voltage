#ifndef __TIMER_UTIL_H__
#define __TIMER_UTIL_H__


#include <time.h>


class TimerUtil
{
private:
    const char *name;
    struct timespec s;

public:
    TimerUtil(const char *name)
    {
        this->name = name;
        clock_gettime(CLOCK_REALTIME, &s);
    };

    ~TimerUtil()
    {
        struct timespec e;
        clock_gettime(CLOCK_REALTIME, &e);
        double elapsed = (e.tv_sec - s.tv_sec) + (e.tv_nsec - s.tv_nsec) / 1e9;
        printf("%s: %.2lf sec\n", name, elapsed);
    };
};


#endif

