#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vector>
#include "malloc_util.h"
#include "TimerUtil.h"
#include "image.h"
#include "motion.h"
#include "shading.h"
#include "signal.h"


int main(int argc, char *argv[])
{
    motion_param_t motion_param;
    motion_param.level = 2;
    motion_param.search_size = 3;
    motion_param.patch_size = 10;
    motion_param.patch_offset = 7;
    motion_param.a_stdev = 1.0;
    motion_param.m_stdev = 3.0;
    motion_param.thresh_xy = 1.0;
    motion_param.length = 2000;
    motion_param.thresh_c = 0.4;

    shading_param_t shading_param;
    shading_param.period = 1000;

    signal_param_t signal_param;
    signal_param.normalize = false;
    signal_param.downsample = true;
    signal_param.period = 100;
    signal_param.patch_size = 8;
    signal_param.patch_offset = 1;

    if(argc != 3)
    {
        printf("%s in.tiff out_path\n", argv[0]);
        return 0;
    }
    char in_file[512], out_path[512];
    strcpy(in_file, argv[1]);
    strcpy(out_path, argv[2]);


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

    tu = new TimerUtil("shading correction");
    correct_shading(shading_param, t, w, h, img, motion_list);
    delete tu;
    
    tu = new TimerUtil("signal extraction");
    int num_out;
    float ***out = extract_signal(signal_param, t, w, h, img, motion_list, range, &num_out);
    delete tu;
    
    tu= new TimerUtil("saving tiff");
    char out_file[512];
    sprintf(out_file, "%s/corrected.tif", out_path);
    bool ret1 = write_tiff(out_file, t, w, h, img);

    sprintf(out_file, "%s/signal.tif", out_path);
    bool ret2 = write_tiff(out_file, num_out, w, h, out);
    delete tu;
    
    free_float3d(img);
    free_float3d(out);
    return (ret1 && ret2) ? 0 : 1;
}

