#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <omp.h>
#include "malloc_util.h"


#define UPPER_BOUND (1.0)
#define LOWER_BOUND (0.0)
#define CLAMP(X) (X < LOWER_BOUND ? LOWER_BOUND : (X > UPPER_BOUND ? UPPER_BOUND : X))


static void compute_derivatives_c(int t, int p, int n,
                                  double **y, double **c, double **z,
                                  double **dc)
{
    #pragma omp parallel for
    for(int k = 0; k < t; k++)
    {
        for(int i = 0; i < n; i++) dc[i][k] = 0;
        
        for(int j = 0; j < p; j++)
        {
            double prod = 1.0;
            for(int i = 0; i < n; i++)
            {
                prod *= 1.0 - c[i][k] * z[i][j];
            }

            for(int i = 0; i < n; i++)
            {
                double prod_1 = 1.0;
                for(int m = 0; m < n; m++)
                {
                    if(m != i) prod_1 *= 1.0 - c[m][k] * z[m][j];
                }
                dc[i][k] -= (y[k][j] - 1 + prod) * prod_1 * z[i][j];
                // below should not be used to avoid zero division
                // dc[i][k] -= (y[j][k] - 1 + prod) * prod * z[j][i] / (1.0 - c[i][k] * z[j][i]);
            }
        }
    }
}

static void compute_derivatives_z(int t, int p, int n,
                                  double **y, double **c, double **z,
                                  double **dz)
{
    #pragma omp parallel for
    for(int j = 0; j < p; j++)
    {
        for(int i = 0; i < n; i++) dz[i][j] = 0;
        
        for(int k = 0; k < t; k++)
        {
            double prod = 1.0;
            for(int i = 0; i < n; i++)
            {
                prod *= 1.0 - c[i][k] * z[i][j];
            }

            for(int i = 0; i < n; i++)
            {
                double prod_1 = 1.0;
                for(int m = 0; m < n; m++)
                {
                    if(m != i) prod_1 *= 1.0 - c[m][k] * z[m][j];
                }
                dz[i][j] -= (y[k][j] - 1 + prod) * prod_1 * c[i][k];
                // below should not be used to avoid zero division
                // dz[j][i] -= (y[j][k] - 1 + prod) * prod * c[i][k] / (1.0 - c[i][k] * z[j][i]);
            }
        }
    }
}

static double compute_error(int t, int p, int n,
                            double **y, double **c, double **z)
{
    double sum = 0;
    for(int k = 0; k < t; k++)
    for(int j = 0; j < p; j++)
    {
        double prod = 1.0;
        for(int i = 0; i < n; i++)
        {
            prod *= 1.0 - c[i][k] * z[i][j];
        }
        double dif = y[k][j] - 1.0 + prod;
        sum += dif * dif;
    }
    return sqrt(sum / (t * p));
}

int demix_cells_cpu(int num_frames, int height, int width,
                    double *probability_maps,
                    int num_cells, double *z_init,
                    double **z_out, double **c_out, double *err,
                    int max_iter, double update_step, double iter_thresh,
                    int num_threads)
{
    if(num_threads > 0)
    {
        omp_set_num_threads(num_threads);
    }

    // set up variables
    const int p = width * height; // # pixels
    const int t = num_frames;
    const int n = num_cells;
    double **y = malloc_double2d(t, p);
    double **z = malloc_double2d(n, p);
    double **c = malloc_double2d(n, t);
    double **dz = malloc_double2d(n, p); // derivative
    double **dc = malloc_double2d(n, t); // derivative
    
    for(int k = 0; k < t; k++)
    for(int j = 0; j < p; j++)
    {
        y[k][j] = probability_maps[k * p + j];
    }

    for(int i = 0; i < n; i++)
    for(int k = 0; k < t; k++)
    {
        c[i][k] = 0.5;
    }

    for(int i = 0; i < n; i++)
    for(int j = 0; j < p; j++)
    {
        z[i][j] = z_init[i * p + j];
    }

    int num_iter = 0;
    double update_norm = iter_thresh + 1.0;
    while(num_iter < max_iter && update_norm > iter_thresh)
    {
        compute_derivatives_c(t, p, n, y, c, z, dc);
        //double c_dif = 0;
        #pragma omp parallel for
        for(int k = 0; k < t; k++)
        for(int i = 0; i < n; i++)
        {
            double c_new = CLAMP(c[i][k] - update_step * dc[i][k]);
            //c_dif += (c_new - c[i][k]) * (c_new - c[i][k]);
            c[i][k] = c_new;
        }
        //c_dif = sqrt(c_dif / (n * t)) / update_step;

        compute_derivatives_z(t, p, n, y, c, z, dz);
        double z_dif = 0;
        #pragma omp parallel for reduction(+: z_dif)
        for(int j = 0; j < p; j++)
        for(int i = 0; i < n; i++)
        {
            double z_new = CLAMP(z[i][j] - update_step * dz[i][j]);
            z_dif += (z_new - z[i][j]) * (z_new - z[i][j]);
            z[i][j] = z_new;
        }
        z_dif = sqrt(z_dif / (n * p)) / update_step;
        
        num_iter++;
        update_norm = z_dif;
    }
    
    *err = compute_error(t, p, n, y, c, z);
    
    // output
    *z_out = z[0];
    *c_out = c[0];
    free(z);
    free(c);

    free_double2d(y);
    free_double2d(dz);
    free_double2d(dc);
        
    return num_iter;
}

