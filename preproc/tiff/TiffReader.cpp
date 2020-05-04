#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <tiffio.h>
#include "TiffReader.h"


void TiffReader::init()
{
    tif = NULL;
    buf = NULL;
    row = 0;
    current_page = 0;
}

TiffReader::TiffReader()
{
    init();
}

TiffReader::~TiffReader()
{
    close();
}

bool TiffReader::open(char *filename)
{
    if((tif = TIFFOpen(filename, "r")) == NULL)
    {
        fprintf(stderr, "failed to open: %s\n", filename);
        return false;
    }

    num_pages = 0;
    do
    {
        uint32 w, h;
        uint16 bps, spp, sfmt;
        TIFFGetField(tif, TIFFTAG_IMAGEWIDTH, &w);
        TIFFGetField(tif, TIFFTAG_IMAGELENGTH, &h);
        TIFFGetField(tif, TIFFTAG_BITSPERSAMPLE, &bps);
        TIFFGetField(tif, TIFFTAG_SAMPLESPERPIXEL, &spp);
        if(TIFFGetField(tif, TIFFTAG_SAMPLEFORMAT, &sfmt) == 0) sfmt = 1;
        
        if(num_pages == 0)
        {
            width = w;
            height = h;
            bits_per_sample = bps;
            samples_per_pixel = spp;
            sample_format = sfmt;
        }
        else
        {
            if((int)w != width || (int)h != height
            || (int)bps != bits_per_sample
            || (int)spp != samples_per_pixel
            || (int)sfmt != sample_format)
            {
                fprintf(stderr, "per-page attributes are not supported\n");
                return false;
            }
        }
        num_pages++;
    }
    while(TIFFReadDirectory(tif));

    if((buf = _TIFFmalloc(TIFFScanlineSize(tif))) == NULL)
    {
        fprintf(stderr, "failed to allocate memory\n");
        return false;
    }
    return true;
}

void TiffReader::close()
{
    if(buf != NULL) _TIFFfree(buf);
    if(tif != NULL) TIFFClose(tif);
    init();
}

void TiffReader::dump()
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

int TiffReader::get_num_pages() { return num_pages; }
int TiffReader::get_width() { return width; }
int TiffReader::get_height() { return height; }
int TiffReader::get_bits_per_sample() { return bits_per_sample; }
int TiffReader::get_samples_per_pixel() { return samples_per_pixel; }
int TiffReader::get_sample_format() { return sample_format; }


bool TiffReader::set_page(int page)
{
    if(page == current_page + 1) // next page
    {
        if(TIFFReadDirectory(tif) == 1) // faster than TIFFSetDirectory()
        {
            current_page++;
            set_row(0);
            return true;
        }
        else return false;
    }
    else
    {
        if(TIFFSetDirectory(tif, page) == 1)
        {
            current_page = page;
            set_row(0);
            return true;
        }
        else return false;
    }
}

void TiffReader::set_row(int row)
{
    this->row = row;
}

void *TiffReader::read_scanline()
{
    if(row >= height)
    {
        bzero(buf, TIFFScanlineSize(tif));
    }
    else if(TIFFReadScanline(tif, buf, row++) < 0)
    {
        return NULL;
    }
    return (void *)buf;
}

bool TiffReader::same_attributes(TiffReader *other)
{
    return (num_pages == other->num_pages)
        && (width == other->width)
        && (height == other->height)
        && (bits_per_sample == other->bits_per_sample)
        && (samples_per_pixel == other->samples_per_pixel)
        && (sample_format == other->sample_format);
}

bool TiffReader::read_page(int page, float **img)
{
    if(!set_page(page)) return false;
    
	int i, j;

    if(samples_per_pixel != 1)
    {
        fprintf(stderr, "multi channel tiff is not supported\n");
        return false;
    }
 
    if(sample_format == 1) // unsigned int
    {	
        if(bits_per_sample == 8)
        {		
		    for(j = 0; j < height; j++)
		    {
			    uint8_t *line = (uint8_t *)read_scanline();
			    for(i = 0; i < width; i++)
			    {
			        img[i][j] = line[i] / (float)0xff;
			    }
		    }
		}
        else if(bits_per_sample == 16)
        {		
		    for(j = 0; j < height; j++)
		    {
			    uint16_t *line = (uint16_t *)read_scanline();
			    for(i = 0; i < width; i++)
			    {
			        img[i][j] = line[i] / (float)0xffff;
			    }
		    }
		}
		else if(bits_per_sample == 32)
		{    
		    for(j = 0; j < height; j++)
		    {
			    uint32_t *line = (uint32_t *)read_scanline();
			    for(i = 0; i < width; i++)
			    {
			        img[i][j] = line[i] / (float)0xffffffff;
			    }
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
		    for(j = 0; j < height; j++)
		    {
			    float *line = (float *)read_scanline();
			    for(i = 0; i < width; i++)
			    {
			        img[i][j] = line[i];
			    }
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


