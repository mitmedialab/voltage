#ifndef __DEMIX_H__
#define __DEMIX_H__

/*

C implementation of demix_cells_with_given_number() function
with OpenMP parallelization on CPU.

*/
double demix_cells_cpu(int num_frames, int height, int width,
                       double *probability_maps,
                       int num_cells, double *z_init,
                       double **z_out, double **c_out,
                       int max_iter, double update_step, double iter_thresh,
                       int num_threads);

#endif

