#ifndef __TIFF_READER_H__
#define __TIFF_READER_H__


#include <tiffio.h>


class TiffReader
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
    int current_page;

public:
    TiffReader();
    ~TiffReader();
    bool open(char *filename);
    void close();
    void dump();
    int get_num_pages();
    int get_width();
    int get_height();
    int get_bits_per_sample();
    int get_samples_per_pixel();
    int get_sample_format();
    
private:
    bool set_page(int page);
    void set_row(int row);
    void *read_scanline();

public:
    bool same_attributes(TiffReader *other);
    bool read_page(int page, float **img);

private:
    void init();
};


#endif

