#ifndef __KALMAN_FILTER_H__
#define __KALMAN_FILTER_H__


class KalmanFilter
{
private:
    // state variables
    float x; // position
    float v; // velocity
    float x_var; // estimate variance of x
    float v_var; // estimate variance of v
    float xv_cov; // estimate covariance of x and v

    // model parameters
    float acceleration_variance;
    float measurement_variance;
    float dt; // time step
    float dt2, dt3, dt4; // power of time step

public:
    KalmanFilter(float a_var, float m_var, float time_step);
    void initialize(float x, float v, float x_var, float v_var);
    float update(float x);
};


#endif

