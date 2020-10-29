#include <map>
#include <string>

#ifndef __WRAPPER_HPP__
#define __WRAPPER_HPP__

#include "coord_descent.hpp"

void CoordDescent_fit(float *X, float *W, float *H, int num_samples, int num_features, int n_components, int max_iterations, double tolerance, int *n_iter);

#endif