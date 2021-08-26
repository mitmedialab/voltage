#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <omp.h>


#define UPPER_BOUND (1.0)
#define LOWER_BOUND (0.0)
#define CLAMP(X) (X < LOWER_BOUND ? LOWER_BOUND : (X > UPPER_BOUND ? UPPER_BOUND : X))


static double **malloc_double2d(int w, int h)
{
    double **buf;
    if((buf = (double **)malloc(sizeof(double *) * w)) == NULL)
    {
        fprintf(stderr, "failed to allocate memory\n");
        return NULL;
    }
    if((buf[0] = (double *)malloc(sizeof(double) * w * h)) == NULL)
    {
        fprintf(stderr, "failed to allocate memory\n");
        free(buf);
        return NULL;
    }
    for(int i = 1; i < w; i++) buf[i] = buf[0] + i * h;
    return buf;
}

static void free_double2d(double **buf)
{
    free(buf[0]);
    free(buf);
}                         



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

double demix_cells_cpu(int num_frames, int height, int width,
                       double *probability_maps,
                       int num_cells, double *z_init,
                       double **z_out, double **c_out,
                       int max_iter, double update_step, double iter_thresh,
                       int num_threads)
{
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
        c[i][k] = rand() / (double)RAND_MAX;
    }

    for(int i = 0; i < n; i++)
    for(int j = 0; j < p; j++)
    {
        z[i][j] = z_init[i * p + j];
    }

    omp_set_num_threads(num_threads);
    int m;
    for(m = 0; m < max_iter; m++)
    {
        compute_derivatives_c(t, p, n, y, c, z, dc);
        double sum_dif_c = 0;
        for(int i = 0; i < n; i++)
        for(int k = 0; k < t; k++)
        {
            double v = CLAMP(c[i][k] - update_step * dc[i][k]);
            sum_dif_c += (v - c[i][k]) * (v - c[i][k]);
            c[i][k] = v;
        }
        sum_dif_c = sqrt(sum_dif_c / (n * t)) / update_step;

        compute_derivatives_z(t, p, n, y, c, z, dz);
        double sum_dif_z = 0;
        for(int i = 0; i < n; i++)
        for(int j = 0; j < p; j++)
        {
            double v = CLAMP(z[i][j] - update_step * dz[i][j]);
            sum_dif_z += (v - z[i][j]) * (v - z[i][j]);
            z[i][j] = v;
        }
        sum_dif_z = sqrt(sum_dif_z / (n * p)) / update_step;
        
        if(sum_dif_z < iter_thresh) break;
    }
    
    double err = compute_error(t, p, n, y, c, z);
    printf("%d cells: %d iterations with error %e\n", num_cells, m, err);
    
    // output
    *z_out = z[0];
    *c_out = c[0];
    free(z);
    free(c);

    free_double2d(y);
    free_double2d(dz);
    free_double2d(dc);
        
    return err;
}

