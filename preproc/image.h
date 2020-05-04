#ifndef __IMAGE_H__
#define __IMAGE_H__


float ***read_tiff(char *filename, int *num_pages, int *width, int *height);
bool write_tiff(char *filename, int num_pages, int width, int height, float ***img);

void normalize_intensity(int num_pages, int width, int height, float ***img);


#endif

