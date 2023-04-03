#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <tiffio.h>

#define uint32 unsigned long

int main(void)
{
    uint32_t width;
    uint32_t height;
    TIFF* img = TIFFOpen("../../Data/three_band/6010_0_0.tif", "r");

    if (img) {
        TIFFGetField(img, TIFFTAG_IMAGEWIDTH, &width);
        TIFFGetField(img, TIFFTAG_IMAGELENGTH, &height);

        // uint32 npixels = width * height;
        // uint32 * raster = (uint32 *) _TIFFmalloc(npixels *sizeof(uint32));

        std::cout << "Width: " << width << std::endl;
        std::cout << "Height: " << height << std::endl;
        TIFFClose(img);
    } else {
        std::cout << "ERROR: Could not load image" << std::endl;
    }

    return 0;
}