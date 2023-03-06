#include <iostream>
#include <fstream>
#include <stdio.h>
#include <stdlib.h>
#include <tiffio.h>
#include <vector>
#include <string>
#include <sstream>
#include <array>
#include <stdbool.h>
#include <cstdlib>

void err_check(bool is_err, char* err_msg);
void start_process(std::vector<std::string> fnames);
void conv_tiff(TIFF* img, const float (&kernel)[3][3]);


int main(int argc, char* argv[]) {
    bool is_err = false; 

    // Check that at least one argument is given.
    if (argc < 2) {
        is_err = true;
    }
    err_check(is_err, "ERROR: No imput files.");

    // Initialize vector with filenames.
    std::vector<std::string> fnames;
    std::ifstream file_point;
    for (int i = 1; i < argc; i++) {
        file_point.open(argv[i]);
        if (file_point) {
            fnames.push_back(std::string(argv[i]));
        }
        file_point.close();
    }

    start_process(fnames);
    
    return 0;
}

void err_check(bool is_err, char* err_msg){
    if (is_err == true) {
        std::cerr << err_msg << std::endl;
        exit(1);
    };    
};

void start_process(std::vector<std::string> fnames){
    // Initialize kernel
    const float kernel[3][3] = {
        {-1.0, -1.0, -1.0},
        {-1.0, 8.0, -1.0},
        {-1.0, -1.0, -1.0}
    };

    // uint32_t columns, rows, npixels;

    // Load each file and send them to be convoluted
    for (int j = 0; j < fnames.size(); j++) {
        TIFF* img = TIFFOpen(fnames[j].c_str(), "r");
        // TIFFGetField(img, TIFFTAG_IMAGELENGTH, &rows);
        // std::cout << "Tiff " << fnames[j] << " length: " << rows << std::endl;
        conv_tiff(img, kernel);
        TIFFClose(img);
    }
};

void conv_tiff(TIFF* tiff, const float (&kernel)[3][3]) {
    uint32_t columns, rows;

    TIFFGetField(tiff, TIFFTAG_IMAGELENGTH, &rows);
    TIFFGetField(tiff, TIFFTAG_IMAGEWIDTH, &columns);

    uint32_t* inbuff = (uint32_t*) _TIFFmalloc(rows * columns *sizeof(uint32_t));
    uint32_t* outbuff = (uint32_t*) _TIFFmalloc(rows * columns *sizeof(uint32_t));

    TIFFReadRGBAImage(tiff, columns, rows, inbuff);
    
    // Perform convolution
    for (int ch = 0; ch < 3; ch++)
        for (int r = 1; r < rows-1; r++) {
            for (int c = 1; c < columns-1; c++) {
                if (c == r) {
                    std::cout << inbuff[r] << std::endl;
                    // std::cout << "r = " << r << "; c = " << c << std::endl;
                }
            }
        }
}