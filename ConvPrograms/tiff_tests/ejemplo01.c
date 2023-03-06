/*
 * Archivo para verificar la versi'on de la liber'ia 
 * hay dos formas
*/
#include <iostream>
#include <stdio.h>
// #include "/opt/homebrew/Cellar/libtiff/4.4.0_1/include/tiffio.h"
#include <tiff.h>
#include <tiffio.h>
#include <tiffconf.h>
#include <tiffio.hxx>
#include <tiffvers.h>

int main(void)
{
	printf("Hola libtiff, la version que uso: %d\n", TIFFLIB_VERSION);
	printf("%s\n", TIFFGetVersion());

	return 0;
}
