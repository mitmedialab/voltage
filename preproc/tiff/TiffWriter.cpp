#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <tiffio.h>
#include "TiffWriter.h"


static void set_fields(TIFF *tif, int width, int height, int bits_per_sample, int samples_per_pixel, int sample_format)
{
    TIFFSetField(tif, TIFFTAG_IMAGEWIDTH, width);
    TIFFSetField(tif, TIFFTAG_IMAGELENGTH, height);
    TIFFSetField(tif, TIFFTAG_BITSPERSAMPLE, bits_per_sample);
    TIFFSetField(tif, TIFFTAG_SAMPLESPERPIXEL, samples_per_pixel);
    TIFFSetField(tif, TIFFTAG_SAMPLEFORMAT, sample_format);

    TIFFSetField(tif, TIFFTAG_ORIENTATION, ORIENTATION_TOPLEFT);
    TIFFSetField(tif, TIFFTAG_PLANARCONFIG, PLANARCONFIG_CONTIG);
    //TIFFSetField(tif, TIFFTAG_COMPRESSION, COMPRESSION_DEFLATE);
    if(samples_per_pixel == 1)
    {
        TIFFSetField(tif, TIFFTAG_PHOTOMETRIC, PHOTOMETRIC_MINISBLACK);
    }
    else if(samples_per_pixel == 3)
    {
        TIFFSetField(tif, TIFFTAG_PHOTOMETRIC, PHOTOMETRIC_RGB);
    }
    else
    {
        fprintf(stderr, "unsupported sample per pixel: %d\n", samples_per_pixel);
        exit(1);
    }
    TIFFSetField(tif, TIFFTAG_ROWSPERSTRIP, TIFFDefaultStripSize(tif, 0));
}


void TiffWriter::init(int width, int height, int bits_per_sample, int samples_per_pixel, int sample_format)
{
    tif = NULL;
    buf = NULL;
    num_pages = 0;
    this->width = width;
    this->height = height;
    this->bits_per_sample = bits_per_sample;
    this->samples_per_pixel = samples_per_pixel;
    this->sample_format = sample_format;
    row = 0;
    first_page = true;
}

TiffWriter::TiffWriter()
{
    init();
}

TiffWriter::~TiffWriter()
{
    close();
}

bool TiffWriter::open(char *filename, int width, int height, int bits_per_sample, int samples_per_pixel, int sample_format, bool is_big_tiff)
{
    init(width, height, bits_per_sample, samples_per_pixel, sample_format);

    if(is_big_tiff)
    {
        tif = TIFFOpen(filename, "w8"); // '8' for BigTIFF to allow >4GB file size
    }
    else
    {
        tif = TIFFOpen(filename, "w");
    }
    
    if(tif == NULL)
    {
        fprintf(stderr, "failed to open: %s\n", filename);
        return false;
    }

    set_fields(tif, width, height, bits_per_sample, samples_per_pixel, sample_format);
    num_pages = 1;
    row = 0;

    if((buf = _TIFFmalloc(TIFFScanlineSize(tif))) == NULL)
    {
        fprintf(stderr, "failed to allocate memory\n");
        return false;
    }
    return true;
}

void TiffWriter::close()
{
    if(buf != NULL) _TIFFfree(buf);
    if(tif != NULL) TIFFClose(tif);
    init();
}

void TiffWriter::dump()
{
    printf("%d pages, %d x %d, %d bits/sample, %d samples/pixel\n", num_pages, width, height, bits_per_sample, samples_per_pixel);
    printf("sample format: ");
    switch(sample_format)
    {
    case 1:
        printf("unsigned integer\n");
        break;
    case 2:
        printf("two's complement signed integer\n");
        break;
    case 3:
        printf("IEEE floating point\n");
        break;
    default:
        printf("undefined\n");
        break;
    }
}

int TiffWriter::get_num_pages() { return num_pages; }
int TiffWriter::get_width() { return width; }
int TiffWriter::get_height() { return height; }
int TiffWriter::get_bits_per_sample() { return bits_per_sample; }
int TiffWriter::get_samples_per_pixel() { return samples_per_pixel; }
int TiffWriter::get_sample_format() { return sample_format; }

bool TiffWriter::add_page()
{
    if(TIFFWriteDirectory(tif) != 1) // this gets extremely slow for large # pages
    {
        fprintf(stderr, "failed to add directory\n");
        return false;
    }
    num_pages++;
    set_fields(tif, width, height, bits_per_sample, samples_per_pixel, sample_format);
    row = 0;
    return true;
}

void TiffWriter::set_row(int row)
{
    this->row = row;
}

tdata_t TiffWriter::get_scanline() { return buf; }

bool TiffWriter::write_scanline()
{
    if(row >= height)
    {
        fprintf(stderr, "number of scanlines (%d) exceeded height (%d)\n", row+1, height);
        return false;
    }
    else if(TIFFWriteScanline(tif, buf, row++, 0) < 0)
    {
        fprintf(stderr, "error writing scanline\n");
        return false;
    }
    return true;
}

bool TiffWriter::write_page(float **img)
{
    if(first_page)
    {
        first_page = false;
    }
    else
    {
        add_page();
    }

	int i, j;

    if(sample_format == 1) // unsigned int
    {	
        if(bits_per_sample == 8)
        {		
		    for(j = 0; j < height; j++)
		    {
			    uint8_t *line = (uint8_t *)get_scanline();
			    for(i = 0; i < width * samples_per_pixel; i++)
			    {
			        line[i] = (uint8_t)(img[i][j] * (float)0xff);
			    }
			    write_scanline();
		    }
		}
        else if(bits_per_sample == 16)
        {		
		    for(j = 0; j < height; j++)
		    {
			    uint16_t *line = (uint16_t *)get_scanline();
			    for(i = 0; i < width * samples_per_pixel; i++)
			    {
			        line[i] = (uint16_t)(img[i][j] * (float)0xffff);
			    }
			    write_scanline();
		    }
		}
		else if(bits_per_sample == 32)
		{    
		    for(j = 0; j < height; j++)
		    {
			    uint32_t *line = (uint32_t *)get_scanline();
			    for(i = 0; i < width * samples_per_pixel; i++)
			    {
			        line[i] = (uint32_t)(img[i][j] * (float)0xffffffff);
			    }
			    write_scanline();
		    }
		}
		else
		{
    	    fprintf(stderr, "unsupported bits per sample: %d\n", bits_per_sample);
    	    return false;
    	}
	}
	else if(sample_format == 2) // signed int
	{
        fprintf(stderr, "signed int unsupported\n");
        return false;
    }
    else if(sample_format == 3) // float
    {
		if(bits_per_sample == 32)
		{
		    if(samples_per_pixel != 1)
            {
                fprintf(stderr, "multi channel float tiff is not supported\n");
                return false;
            }

		    for(j = 0; j < height; j++)
		    {
			    float *line = (float *)get_scanline();
			    for(i = 0; i < width; i++)
			    {
			        line[i] = img[i][j];
			    }
			    write_scanline();
		    }
		}
		else
		{
    	    fprintf(stderr, "unsupported format\n");
    	    return false;
    	}
    }
    else
    {
        fprintf(stderr, "unsupported format\n");
        return false;
    }
	return true;
}

