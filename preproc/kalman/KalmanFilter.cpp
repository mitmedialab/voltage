#include "KalmanFilter.h"

KalmanFilter::KalmanFilter(float a_var, float m_var, float time_step)
{
    acceleration_variance = a_var;
    measurement_variance = m_var;
    dt = time_step;
    dt2 = dt * dt;
    dt3 = dt2 * dt;
    dt4 = dt3 * dt;
}

void KalmanFilter::initialize(float x, float v, float x_var, float v_var)
{
    this->x = x;
    this->v = v;
    this->x_var = x_var;
    this->v_var = v_var;
    this->xv_cov = 0;
}

float KalmanFilter::update(float x_measured)
{
    // prediction
    float x_p = x + v * dt;
    float v_p = v;
    
    float x_var_p  = x_var + 2 * xv_cov * dt + v_var * dt2;
    float xv_cov_p = xv_cov + v_var * dt;
    float v_var_p  = v_var;
    x_var_p  += 0.25 * dt4 * acceleration_variance;
    xv_cov_p += 0.5  * dt3 * acceleration_variance;
    v_var_p  +=        dt2 * acceleration_variance;
    
    // update
    float residual = x_measured - x_p;
    float residual_variance = x_var_p + measurement_variance;
    float gain_x = x_var_p / residual_variance;
    float gain_v = xv_cov_p / residual_variance;
    
    x = x_p + gain_x * residual;
    v = v_p + gain_v * residual;
    
    x_var = x_var_p * (1 - gain_x);
    xv_cov = xv_cov_p * (1 - gain_x);
    v_var = v_var_p - xv_cov_p * gain_v;
    
    return x;
}

