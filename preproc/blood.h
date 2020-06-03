#ifndef __BLOOD_H__
#define __BLOOD_H__


typedef struct
{
    int period; // time period [frames] for temporal spectral analysis
    float frames_per_sec; // video frame rate
    float freq_min; // minimum temporal frequency [Hz] of blood flow
    float freq_max; // maximum temporal frequency [Hz] of blood flow
    float thresh; // threshold to determine if blood or not (<= 1)
} blood_param_t;


float ***suppress_blood(blood_param_t &param, int num_pages, int width, int height, float ***img);


#endif

