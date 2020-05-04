#ifndef __TIFF_WRITER_H__
#define __TIFF_WRITER_H__


#include <tiffio.h>


class TiffWriter
{
private:
    TIFF *tif;
    tdata_t buf;
    int num_pages;
    int width;
    int height;
    int bits_per_sample;
    int samples_per_pixel;
    int sample_format;
    int row;
    bool first_page;

public:
    TiffWriter();
    ~TiffWriter();
    bool open(char *filename, int width, int height, int bits_per_sample, int samples_per_pixel, int sample_format, bool is_big_tiff = false);
    void close();
    void dump();
    int get_num_pages();
    int get_width();
    int get_height();
    int get_bits_per_sample();
    int get_samples_per_pixel();
    int get_sample_format();
    
private:
    bool add_page();
    void set_row(int row);
    tdata_t get_scanline();
    bool write_scanline();

public:
    bool write_page(float **img);

private:
    void init(int width = 0, int height = 0, int bits_per_sample = 0, int samples_per_pixel = 0, int sample_format = 0);
};


#endif

