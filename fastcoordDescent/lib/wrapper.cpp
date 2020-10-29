#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vector>
#include <iostream>

#include "wrapper.hpp"

void CoordDescent_fit(float *X, float *W, float *H, int num_samples, int num_features, int n_components, int max_iterations, double tolerance, int *n_iter)
{
	*n_iter = coordinate_descent(X, W, H, num_samples, num_features, n_components, max_iterations, tolerance);
}