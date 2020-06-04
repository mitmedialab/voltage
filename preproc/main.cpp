#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vector>
#include "malloc_util.h"
#include "TimerUtil.h"
#include "image.h"
#include "motion.h"
#include "shading.h"
#include "blood.h"
#include "signal.h"


#define OPTCMP1(opt) (strcmp(argv[n], opt) == 0)
#define OPTCMP2(opt) (strcmp(argv[n], opt) == 0 && n+1 < argc)


static void print_help(char *command, float frames_per_sec,
                       motion_param_t  &motion_param,
                       shading_param_t &shading_param,
                       blood_param_t   &blood_param,
                       signal_param_t  &signal_param)
{
    printf("%s [options] in.tiff out_path\n", command);
    printf("\n");
    printf("  -ds : disable shading correction\n");
    printf("  -db : disable blood suppression\n");
    printf("\n");
    printf("  -fr <float>: frame rate (%.1f Hz)\n", frames_per_sec);
    printf("\n");
    printf("  motion correction parameters\n");
    printf("  -ms <int>  : search size (%d pixels)\n", motion_param.search_size);
    printf("  -mp <int>  : patch size (%d pixels)\n", motion_param.patch_size);
    printf("  -mo <int>  : patch offset (%d pixels)\n", motion_param.patch_offset);
    printf("\n");
    printf("  shading correction parameters\n");
    printf("  -hw <int>  : window length (%d frames)\n", shading_param.period);
    printf("\n");
    printf("  blood suppression parameters\n");
    printf("  -bw <int>  : window length (%d frames)\n", blood_param.period);
    printf("  -bl <float>: minimum frequency (%.1f Hz)\n", blood_param.freq_min);
    printf("  -bh <float>: maximum frequency (%.1f Hz)\n", blood_param.freq_max);
    printf("  -bt <float>: threshold (%.1f)\n", blood_param.thresh);
    printf("\n");
    printf("  signal extraction parameters\n");
    printf("  -sm <int>  : method PCA=0, max-median=1 (%d)\n", signal_param.method);
    printf("  -sw <int>  : window length (%d frames)\n", signal_param.period);
    printf("  -sc <float>: cutoff frequency (%.1f Hz)\n", signal_param.freq_max);
    printf("  -sp <int>  : PCA patch size (%d pixels)\n", signal_param.patch_size);
    printf("  -so <int>  : PCA patch offset (%d pixels)\n", signal_param.patch_offset);
    printf("  -ss <float>: max-median spatial smoothing (%.1f pixels)\n", signal_param.smooth_scale);
    exit(0);
}

int main(int argc, char *argv[])
{
    motion_param_t motion_param;
    motion_param.level = 2;
    motion_param.search_size = 3;
    motion_param.patch_size = 10;
    motion_param.patch_offset = 7;
    motion_param.x_range = 0.7;
    motion_param.y_range = 1.0;
    motion_param.a_stdev = 1.0;
    motion_param.m_stdev = 3.0;
    motion_param.thresh_xy = 1.0;
    motion_param.length = 2000;
    motion_param.thresh_c = 0.4;

    shading_param_t shading_param;
    shading_param.period = 1000;

    blood_param_t blood_param;
    blood_param.period = 1000;
    blood_param.freq_min = 20.0;
    blood_param.freq_max = 200.0;
    blood_param.thresh = 0.9;

    signal_param_t signal_param;
    signal_param.method = 0;
    signal_param.period = 100;
    signal_param.freq_max = 100.0;
    signal_param.normalize = false;
    signal_param.downsample = true;
    signal_param.patch_size = 8;
    signal_param.patch_offset = 1;
    signal_param.smooth_scale = 2.0;

    bool skip_shading_correction = false;
    bool skip_blood_suppression = false;
    float frames_per_sec = 1000.0;

    char in_file[512], out_path[512];
    int n = 1;
    int m = 0;
    while(n < argc)
    {
        if(     OPTCMP1("-ds")) { skip_shading_correction = true; n++; }
        else if(OPTCMP1("-db")) { skip_blood_suppression = true; n++; }
        else if(OPTCMP2("-fr")) { frames_per_sec = (float)atof(argv[n+1]); n+=2; }
        else if(OPTCMP2("-ms")) { motion_param.search_size  = atoi(argv[n+1]); n+=2; }
        else if(OPTCMP2("-mp")) { motion_param.patch_size   = atoi(argv[n+1]); n+=2; }
        else if(OPTCMP2("-mo")) { motion_param.patch_offset = atoi(argv[n+1]); n+=2; }
        else if(OPTCMP2("-hw")) { shading_param.period = atoi(argv[n+1]); n+=2; }
        else if(OPTCMP2("-bw")) { blood_param.period = atoi(argv[n+1]); n+=2; }
        else if(OPTCMP2("-bl")) { blood_param.freq_min = (float)atof(argv[n+1]); n+=2; }
        else if(OPTCMP2("-bh")) { blood_param.freq_max = (float)atof(argv[n+1]); n+=2; }
        else if(OPTCMP2("-bt")) { blood_param.thresh = (float)atof(argv[n+1]); n+=2; }
        else if(OPTCMP2("-sm")) { signal_param.method = atoi(argv[n+1]); n+=2; }
        else if(OPTCMP2("-sw")) { signal_param.period = atoi(argv[n+1]); n+=2; }
        else if(OPTCMP2("-sc")) { signal_param.freq_max = (float)atof(argv[n+1]); n+=2; }
        else if(OPTCMP2("-sp")) { signal_param.patch_size = atoi(argv[n+1]); n+=2; }
        else if(OPTCMP2("-so")) { signal_param.patch_offset = atoi(argv[n+1]); n+=2; }
        else if(OPTCMP2("-ss")) { signal_param.smooth_scale = (float)atof(argv[n+1]); n+=2; }
        else if(m == 0) { strcpy(in_file,  argv[n]); n++; m++; }
        else if(m == 1) { strcpy(out_path, argv[n]); n++; m++; }
        else { m = 0; break; }
    }
    if(m < 2)
    {
        print_help(argv[0], frames_per_sec,
                   motion_param, shading_param, blood_param, signal_param);
    }
    blood_param.frames_per_sec = frames_per_sec;
    signal_param.frames_per_sec = frames_per_sec;


    TimerUtil *tu;
    
    tu = new TimerUtil("loading tiff");
    int t, w, h;
	float ***img = read_tiff(in_file, &t, &w, &h);
	if(img == NULL) exit(1);
	delete tu;

    // hack: eliminate black line at the bottom
	for(int i = 0; i < t; i++)
	{
	    for(int j = 0; j < w; j++) img[i][j][h-1] = img[i][j][h-2];
	}

    normalize_intensity(t, w, h, img);

    tu = new TimerUtil("motion correction");
    motion_range_t range;
	std::vector<motion_t> motion_list = correct_motion(motion_param, t, w, h, img, range);
	if(motion_list.empty()) exit(1);
	delete tu;
    printf("(x, y) in [%.1f, %.1f] x [%.1f, %.1f]\n",
           range.min_x, range.max_x, range.min_y, range.max_y);

    if(skip_shading_correction)
    {
        printf("shading correction skipped\n");
    }
    else
    {
        tu = new TimerUtil("shading correction");
        correct_shading(shading_param, t, w, h, img, motion_list);
        delete tu;
    }
    
    float ***bld = NULL;
    if(skip_blood_suppression)
    {
        printf("blood suppression skipped\n");
    }
    else
    {
        tu = new TimerUtil("blood suppression");
        bld = suppress_blood(blood_param, t, w, h, img);
        delete tu;
    }
    
    tu = new TimerUtil("signal extraction");
    int num_out;
    float ***out = extract_signal(signal_param, t, w, h, img, motion_list, range, &num_out);
    delete tu;
    
    tu = new TimerUtil("saving tiff");
    char out_file[512];
    sprintf(out_file, "%s/corrected.tif", out_path);
    bool ret_c = write_tiff(out_file, t, w, h, img);

    sprintf(out_file, "%s/blood.tif", out_path);
    bool ret_b = skip_blood_suppression || write_tiff(out_file, 3 + blood_param.period / 2, w, h, bld);

    sprintf(out_file, "%s/signal.tif", out_path);
    bool ret_s = write_tiff(out_file, num_out, w, h, out);
    delete tu;
    
    free_float3d(img);
    if(bld != NULL) free_float3d(bld);
    free_float3d(out);
    return (ret_c && ret_b && ret_s) ? 0 : 1;
}

