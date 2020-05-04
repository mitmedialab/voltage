#include "malloc_util.h"
#include "TiffReader.h"
#include "TiffWriter.h"

float ***read_tiff(char *filename, int *num_pages, int *width, int *height)
{
    TiffReader *tr = new TiffReader();
    if(!tr->open(filename)) return NULL;
    tr->dump();
    
    const int t = tr->get_num_pages();
    const int w = tr->get_width();
    const int h = tr->get_height();
    
	float ***img = malloc_float3d(t, w, h);
	
	for(int i = 0; i < t; i++)
	{
	    if(!tr->read_page(i, img[i]))
	    {
	        tr->close();
	        delete tr;
	        free_float3d(img);
	        return NULL;
	    }
	}
	tr->close();
    delete tr;
    
    *num_pages = t;
    *width = w;
    *height = h;
    return img;
}

bool write_tiff(char *filename, int num_pages, int width, int height, float ***img)
{
	unsigned long long file_size = (unsigned long long)num_pages * width * height * sizeof(float);
	TiffWriter *tw = new TiffWriter();
	if(!tw->open(filename, width, height, 32, 1, 3, (file_size >> 32))) return false;
	for(int i = 0; i < num_pages; i++)
	{
	    if(!tw->write_page(img[i]))
	    {
	        tw->close();
	        delete tw;
	        return false;
	    }
	}
    tw->close();
    delete tw;
    return true;
}

void normalize_intensity(int num_pages, int width, int height, float ***img)
{
    float *lumi = malloc_float1d(num_pages);
    float max = 0;
    
    for(int k = 0; k < num_pages; k++)
    {
        lumi[k] = 0;
        for(int i = 0; i < width; i++)
        for(int j = 0; j < height; j++)
        {
            lumi[k] += img[k][i][j];
            if(max < img[k][i][j]) max = img[k][i][j];
        }
    }

    #pragma omp parallel for
    for(int k = 0; k < num_pages; k++)
    {
        float scale = lumi[0] / lumi[k] / max;
        for(int i = 0; i < width; i++)
        for(int j = 0; j < height; j++)
        {
            img[k][i][j] *= scale;
        }
    }
}

