#include <iostream>
#include <fstream>
// #include <stdio.h>
// #include <stdlib.h>
#include <tiffio.h>
#include <vector>
// #include <string>
// #include <sstream>
#include <array>
// #include <stdbool.h>
#include <cstdlib>
#include <cmath>

void err_check(bool is_err, char* err_msg);
void start_process(std::vector<std::string> fnames);
void conv_tiff(TIFF* img, const float (&kernel)[9]);
float sigmoid(float x) {
    return (255.0/(1.0 + exp(-x)));
}


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
    const float kernel[9] = {
        -1.0, -1.0, -1.0,
        -1.0, 8.0, -1.0,
        -1.0, -1.0, -1.0
    };

    // uint32_t columns, rows, npixels;

    // Load each file and send them to be convoluted
    for (int j = 0; j < fnames.size(); j++) {
        TIFF* img = TIFFOpen(fnames[j].c_str(), "r");
        conv_tiff(img, kernel);
        TIFFClose(img);
    }
};

void conv_tiff(TIFF* tiff, const float (&kernel)[9]) {
    uint32_t columns, rows; // columns -> number of columns, rows -> number of rows

    TIFFGetField(tiff, TIFFTAG_IMAGELENGTH, &rows);
    TIFFGetField(tiff, TIFFTAG_IMAGEWIDTH, &columns);

    uint32_t* inbuff = (uint32_t*) _TIFFmalloc(rows * columns * sizeof(uint32_t));
    uint32_t* outbuff = (uint32_t*) _TIFFmalloc(rows * columns * sizeof(uint32_t));

    TIFFReadRGBAImage(tiff, columns, rows, inbuff);

    // Arrays for components
    std::vector<uint8_t> r(rows * columns);
    std::vector<uint8_t> g(rows * columns);
    std::vector<uint8_t> b(rows * columns);

    for (int i = 0; i < rows * columns; i++) {
        r[i] = TIFFGetR(inbuff[i]);
        g[i] = TIFFGetG(inbuff[i]);
        b[i] = TIFFGetB(inbuff[i]);
    }

    float sum_r, sum_g, sum_b, sum_a;
    int radius = 1, kernel_size = 3; // Radius of convolution kernel and size of kernel
    int pixel_x, pixel_y, pixel_index;
    float weight;
    uint32_t conv_pixel;

    // Convolution
    for (int j = 0; j < rows; j++) {
        for (int i = 0; i < columns; i++) {
            sum_r = 0; sum_g = 0.0; sum_b = 0.0;
            for (int y = -radius; y <= radius; y++) {
                for (int x = -radius; x <= radius; x++) {
                    pixel_y = j + y;
                    pixel_x = i + x;
                    if (pixel_x >= 0 && pixel_x < columns && pixel_y >= 0 && pixel_y < rows) { // Check that its not over the edge
                        pixel_index = pixel_y * columns + pixel_x;
                        weight = kernel[(y + radius) * kernel_size + (x + radius)];
                        sum_r += r[pixel_index] * weight;
                        sum_g += g[pixel_index] * weight;
                        sum_b += b[pixel_index] * weight;
                    }
                }
            }

            int sum_rc = uint32_t(sigmoid(sum_r));
            int sum_gc = uint32_t(sigmoid(sum_g));
            int sum_bc = uint32_t(sigmoid(sum_b));

            // At this point, we have calculated one of the convoluted picture's pixel. We can proceed to store the calculated values.
            conv_pixel = (sum_rc << 16) | (sum_gc << 8) | sum_bc;

            // Store result in outbuff
            outbuff[j * columns + i] = conv_pixel;

        }
    }

    TIFF* outimg = TIFFOpen("conv_img.tif", "w");
    if (outimg) {
        TIFFSetField(outimg, TIFFTAG_IMAGEWIDTH, columns);
        TIFFSetField(outimg, TIFFTAG_IMAGELENGTH, rows);
        TIFFSetField(outimg, TIFFTAG_SAMPLESPERPIXEL, 3);
        TIFFSetField(outimg, TIFFTAG_BITSPERSAMPLE, 16);
        // TIFFSetField(outimg, TIFFTAG_ORIENTATION, ORIENTATION_TOPLEFT);
        TIFFSetField(outimg, TIFFTAG_PLANARCONFIG, 2);
        TIFFSetField(outimg, TIFFTAG_PHOTOMETRIC, 2);
        TIFFWriteEncodedStrip(outimg, 0, outbuff, columns * rows * sizeof(uint32_t));
        TIFFClose(outimg);
    }

    delete[] inbuff;
    delete[] outbuff;

    std::cout << "Yay!" << std::endl;


}