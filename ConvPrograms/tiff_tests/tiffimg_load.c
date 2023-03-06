/* Archivo para  abrir una imagen e imprimr 
 * detalles importantes  
*/
#include <stdio.h>
#include <stdlib.h>
#include <tiffio.h>
#define imsize 18224

int main(void)
{
	int count;
	int count2;
	uint8* im;
	uint32 imagelength;
	uint32 width;
	im = (uint8*)malloc(imsize*sizeof(uint8));

	TIFF* tif = TIFFOpen("6010_0_0_A.tif","r");
	if (!tif)
	{	
		printf("%p", tif);
		exit(1);
	}

	tsize_t scanline;
	tdata_t buf;
	uint32 row;
	uint32 col;

	uint16 nsamples;

	TIFFGetField(tif, TIFFTAG_SAMPLESPERPIXEL, &nsamples);
	TIFFGetField(tif, TIFFTAG_IMAGELENGTH, &imagelength);
	TIFFGetField(tif, TIFFTAG_IMAGEWIDTH, &width);

	scanline = TIFFScanlineSize(tif);
	buf = _TIFFmalloc(scanline);
	uint8* data;


	for (row = 0; row < imagelength; row++) {
		TIFFReadScanline(tif, buf, row,1);
		count2++;
		data = (uint8*)buf;      

		for (col = 0; col < scanline; col++) {                                       
			printf("%d ", *data);
			im[count] = *data;
			count++;
			data++;                             
		}

		printf("im[1]= %d\n im[2] = %d \n im[3] = %d \n im[18224] = %d\n",im[0],im[1],im[2],im[18224]);
		_TIFFfree(buf);
		TIFFClose(tif);
		free(im);

		printf("num of cols= %d\n",count);
		printf("num of rows = %d\n",count2);//both counts print col size
		printf("width = %d\n",width); //prints row size
		printf("\n");
	}

	TIFFClose(tif);

	return 0;
}
