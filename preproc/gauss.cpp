#include <stdio.h>
#include <stdlib.h>
#include <math.h>


static float s2q(float stdev)
{
    if(stdev < 0.5)
    {
        fprintf(stderr, "too small scale: %f\n", stdev); exit(1);
    }
    else if(stdev < 2.5)
    {
        return 3.97156 - 4.14554 * sqrt(1.0 - 0.26891 * stdev);
    }
    else return 0.98711 * stdev - 0.9633;
}

void recursive_gauss_set_filter(float stdev, float *c)
{
    float q = s2q(stdev);
    float q2 = q * q;
    float q3 = q2 * q;
    
    float b0 = 1.57825 + 2.44413 * q + 1.4281 * q2 + 0.422205 * q3;
    float b1 = 2.44413 * q + 2.85619 * q2 + 1.26661 * q3;
    float b2 = -1.4281 * q2 - 1.26661 * q3;
    float b3 = 0.422205 * q3;

    c[1] = b1 / b0;
    c[2] = b2 / b0;
    c[3] = b3 / b0;
    c[0] = 1.0 - c[1] - c[2] - c[3];
}

void recursive_gauss_apply_filter(float *c, int n, float *dat)
{
    for(int i = 3; i < n; i++)
    {
        dat[i] = c[0] * dat[i] + c[1] * dat[i-1] + c[2] * dat[i-2] + c[3] * dat[i-3];
    }
    for(int i = n - 4; i >= 0; i--)
    {
        dat[i] = c[0] * dat[i] + c[1] * dat[i+1] + c[2] * dat[i+2] + c[3] * dat[i+3];
    }
}

