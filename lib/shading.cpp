#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <vector>
#include "shading.hpp"

double solve_with_qr_decomposition(int row, int col, double **a, double *x);


static float *regress_motion(std::vector<float> &v, std::vector<float> &mx, std::vector<float> &my)
{
    // pixel_val = c0 + c1 mx + c2 my + c3 mx^2 + c4 mx my + c5 my^2
    //
    // fit it to data points by solving min |A c = b|^2
    // where c = [c0 c1 c2 c3 c4 c5]^T is the coefficient vector
    // A is the matrix whose row [1, mx, my, mx^2, mx my, my^2] is given by each data point
    // b is the observation vector v
    //
    // for implementation reason, a[i][j] represents j-th row and i-th column
    // and a[-1][j] represents b[j], j-th row of b

    const int m = (int)v.size();
    const int n = 6;
    double *a = (double *) malloc(m * (n+1) * sizeof(double));
    // malloc_double2d(m + 1, n);

    for(int k = 0; k < m; k++)
    {
        float x = mx[k];
        float y = my[k];
        *loc2D(a, m, (n+1), k, 0) = 1;
        *loc2D(a, m, (n+1), k, 1) = x;
        *loc2D(a, m, (n+1), k, 2) = y;
        *loc2D(a, m, (n+1), k, 3) = x * x;
        *loc2D(a, m, (n+1), k, 4) = x * y;
        *loc2D(a, m, (n+1), k, 5) = y * y;
        *loc2D(a, m, (n+1), k, 6) = v[k];
    }

    double c[6];
    solve_with_qr_decomposition(m, n, a, c);
    free(a);
    
    float *model = (float *) malloc(m * sizeof(float)); 
    for(int k = 0; k < m; k++)
    {
        float x = mx[k];
        float y = my[k];
        //model[k] = (float)(c[0] + c[1] * x + c[2] * y + c[3] * x * x + c[4] * x * y + c[5] * y * y);
        // better to leave out the constant term c[0]
        // as it is the baseline intensity when motion is zero (x = y = 0)
        model[k] = (float)(c[1] * x + c[2] * y + c[3] * x * x + c[4] * x * y + c[5] * y * y);
    }
    return model;
}

void correct_shading(shading_param_t &param,
                     int num_pages, int width, int height, float *img,
                     std::vector<motion_t> motion)
{
	const int period = param.period;

	for(int frame = 0; frame < num_pages; frame += period)
	{
	    std::vector<float> mx, my;
	    for(int k = 0; (k < period) && (frame + k < num_pages); k++)
	    {
	        if(motion[frame + k].valid)
	        {
	            mx.push_back(motion[frame + k].x);
	            my.push_back(motion[frame + k].y);
	        }
	    }
	    
	    const int n = (int)mx.size();
	    if(n < period / 2)
	    {
	        fprintf(stderr, "too few (%d) valid frames at frame %d, skipping\n", n, frame);
	        continue;
	    }

	    // take a look at the motion range during this period
	    // and determine whether or not regression should be performed
	    // because too small range can't reliably model the shading
	    // and there is no point in correcting shading if motion is small	    
	    float avg_x = mx[0];
	    float avg_y = my[0];
  	    float min_x = mx[0];
	    float min_y = my[0];
	    float max_x = mx[0];
	    float max_y = my[0];
	    for(int k = 1; k < n; k++)
	    {
	        avg_x += mx[k];
	        avg_y += my[k];
	        if(min_x > mx[k]) min_x = mx[k];
	        if(min_y > my[k]) min_y = my[k];
	        if(max_x < mx[k]) max_x = mx[k];
	        if(max_y < my[k]) max_y = my[k];
	    }
	    if(max_x - min_x < 1 || max_y - min_y < 1)
	    {
	        fprintf(stderr, "too small motion range [%f, %f] at frame %d, skipping\n",
	                max_x - min_x, max_y - min_y, frame);
	        continue;
	    }

	    // center motion vectors (0 mean) for each period
	    // so that the constant term c[0] will correspond to the center intensity
	    avg_x /= n;
	    avg_y /= n;
	    for(int k = 0; k < n; k++)
	    {
	        mx[k] -= avg_x;
	        my[k] -= avg_y;
	    }

	    #pragma omp parallel for
	    for(int i = 0; i < height; i++) {
	    	for(int j = 0; j < width; j++) {
	    		std::vector<float> v;
	    		for(int k = 0; (k < period) && (frame + k < num_pages); k++) {
	    	        if(motion[frame + k].valid) {
	    	            float f = *loc3D(img, num_pages, height, width, frame + k, i, j);
		                v.push_back(f);
		            }
		        }

		        float *ret = regress_motion(v, mx, my);
		        int idx = 0;
		        for(int k = 0; (k < period) && (frame + k < num_pages); k++)
		        {
	    	        if(motion[frame + k].valid)
		            {
	    	            *loc3D(img, num_pages, height, width, frame + k, i, j) -= ret[idx++]; // subtract intensity change coming from motion
	    	        }
		        }
		        free_float1d(ret);

	    	}
	    }
	}
}

	    
