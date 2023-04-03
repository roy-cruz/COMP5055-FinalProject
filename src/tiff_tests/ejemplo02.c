/* Archivo para  abrir una imagen e imprimr 
 * detalles importantes  
*/
#include <stdio.h>
#include <stdlib.h>
#include <tiffio.h>

int main(void)
{
	TIFF* tif = TIFFOpen("../Data/three_band/6010_0_0.tif", "r");
	if (!tif)
	{	
		printf("%p", tif);
		exit(1);
	}

	uint32_t r;		// row index
	uint32_t w;		// column index
	uint32_t rows;		// number of row in image
	uint32_t columns;		// number of columns in image
	uint16_t bitsPerSample;	// normally 8 for grayscale, 16 for color 
	uint16_t samplesPerPixel;	// normally 1 for grayscale, 3 for color
	uint16_t photoMetric;	// normally 1 for grayscale, 2 for color
	uint16_t planarConf;	// how the component of each pixel are stored
				// 1 for chunky format. For RGB, data is RGBRGBRGB...
				// 2 planar format. Componets are stored in separate
				// component planes. For RGB, color in a plane. 
	uint16_t lineSize;	// get the size bytes of each line


	TIFFGetField(tif, TIFFTAG_IMAGELENGTH, &rows);
	TIFFGetField(tif, TIFFTAG_IMAGEWIDTH, &columns);
	TIFFGetField(tif, TIFFTAG_BITSPERSAMPLE, &bitsPerSample);
	TIFFGetField(tif, TIFFTAG_SAMPLESPERPIXEL, &samplesPerPixel);
	TIFFGetField(tif, TIFFTAG_PHOTOMETRIC, &photoMetric);
	TIFFGetField(tif, TIFFTAG_PLANARCONFIG, &planarConf);

	lineSize = TIFFScanlineSize(tif);

	printf("rows = %d\n", rows);
	printf("columns = %d\n", columns);
	printf("bits per sample = %d\n", bitsPerSample);
	printf("samples per pixel = %d\n", samplesPerPixel);
	printf("photometric = %d\n", photoMetric);
	printf("line size = %d\n", lineSize);
	printf("planar config = %d\n", planarConf);

	
	TIFFClose(tif);

	return 0;
}
