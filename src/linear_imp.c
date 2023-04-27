/* File:     linear_imp.c
 *
 * Purpose:  Perform an edge detection convolution of a series of .tif/.tiff images.
 *
 * Compile:  g++ linear_imp.c -o linear_imp
 * Run:      ./linear_imp <path to TIFF 1> <path to TIFF 2> ...
 *
 * Input:    Path to TIFF images to be covoluted
 * Output:   TIFF images which highlight the edges detected on the input TIFF images.
 *
 * Errors:   If an error is detected (no images provided or no images can be opened), the
 *           program prints a message and all processes quit.
 *
 */
#include <iostream>
#include <tiffio.h>
#include <vector>
#include <cmath>
#include <string>
#include <cstring>
#include <stdbool.h>

void err_check(bool ok, char* err_msg);
void conv_tiff(uint32_t *inbuff, uint32_t *outbuff, int width, int height, const float (&kernel_x)[9], const float (&kernel_y)[9]);
float sigmoid(float x);

int main(int argc, char* argv[]) {

    err_check(argc >= 2, "ERROR: No imput files.");

    // Initialize vector TIFF pointers
    TIFF* tiff = NULL;
    std::vector<TIFF*> tiff_points;
    for (int i = 1; i < argc; i++) {
        tiff = TIFFOpen((argv[i]), "r");
        if (tiff != NULL)
            tiff_points.push_back(tiff);
        else
            std::cout << "WARNING: \'" << argv[i] << "\' could not be opened." << std::endl;
    }


    // Check that at least one tiff was opened.
    err_check(tiff_points.size() != 0, "ERROR: No tiff file could be opened.");

    // Prewitt operators (i.e. kernels) for edge detection (i.e. gradient magnitude calculation)
    const float edge_x[9] = {
        1.0, 0.0, -1.0,
        1.0, 0.0, -1.0,
        1.0, 0.0, -1.0
    };

    const float edge_y[9] = {
        1.0, 1.0, 1.0,
        0.0, 0.0, 0.0,
        -1.0, -1.0, -1.0
    };

    // Convolute all the tiffs! Also close each tiff after processing it.
    for (int j = 0; j < tiff_points.size(); j++) {
        // Get number of rows and number of columns
        uint32_t width, height; 
        TIFFGetField(tiff_points[j], TIFFTAG_IMAGELENGTH, &height);
        TIFFGetField(tiff_points[j], TIFFTAG_IMAGEWIDTH, &width);

        // Make space in memory for input buffer and output buffer.
        uint32_t* inbuff = (uint32_t*) _TIFFmalloc(height * width * sizeof(uint32_t));
        uint32_t* outbuff = (uint32_t*) _TIFFmalloc(height * width * sizeof(uint32_t));

        // Read the image data from tiff and store it in inbuff
        TIFFReadRGBAImage(tiff_points[j], width, height, inbuff, 0);

        // Perform convolution
        conv_tiff(inbuff, outbuff, width, height, edge_x, edge_y);

        TIFFClose(tiff_points[j]);

        std::string fname_str = "_conv.tiff";
        std::string fnameorig (argv[j+1]);
        fnameorig = fnameorig.substr(-13, 8);
        fname_str = "./convoluted/" + fnameorig + fname_str;

        char fname[15];
        std::strcpy(fname, fname_str.c_str());

        // Construct tiff from outbuff.
        TIFF* outimg = TIFFOpen(fname, "w");
        if (outimg) {
            TIFFSetField(outimg, TIFFTAG_IMAGEWIDTH, width);
            TIFFSetField(outimg, TIFFTAG_IMAGELENGTH, height);
            TIFFSetField(outimg, TIFFTAG_BITSPERSAMPLE, 8);
            TIFFSetField(outimg, TIFFTAG_SAMPLESPERPIXEL, 4);
            TIFFSetField(outimg, TIFFTAG_PLANARCONFIG, PLANARCONFIG_CONTIG);
            TIFFSetField(outimg, TIFFTAG_PHOTOMETRIC, PHOTOMETRIC_RGB);

            // Write one scanline at a time
            for (int i = 0; i < height; i ++)
                TIFFWriteScanline(outimg, &outbuff[(height - i - 1) * width], i);

            TIFFClose(outimg);
        }

        delete[] inbuff;
        delete[] outbuff;

        std::cout << "Convoluted file " << fnameorig << ".tiff" << std::endl;
    };
    
    return 0;
}

void err_check(bool ok, char* err_msg){
    if (ok == false) {
        std::cerr << err_msg << std::endl;
        exit(1);
    }
}

void conv_tiff(uint32_t *inbuff, uint32_t *outbuff, int width, int height, const float (&kernel_x)[9], const float (&kernel_y)[9]) {
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            // Gradient value initialization.
            float grad_x = 0.0f;
            float grad_y = 0.0f;

            // Loop over kernel elements
            for (int k = -1; k <= 1; k++) {
                for (int l = -1; l <= 1; l++){
                    // Get pixel coodinates in the input image
                    int x = j + l;
                    int y = i + k;

                    // Clamp the coordinates to the image boundaries
                    x = std::max(0, std::min(x, (int)width - 1));
                    y = std::max(0, std::min(y, (int)height - 1));

                    // Get pixel value in the input image
                    uint32_t pixel = inbuff[y * width + x];

                    uint8_t r = TIFFGetR(pixel);
                    uint8_t g = TIFFGetG(pixel);
                    uint8_t b = TIFFGetB(pixel);

                    // Convert pixel value to grayscale
                    float gray = 0.2126f * r + 0.7152f * g + 0.0722f * b;

                    // Multiply pixel value with corresponding kernel element and add to the gradient values
                    grad_x += gray * kernel_x[(k + 1) * 3 + (l + 1)];
                    grad_y += gray * kernel_y[(k + 1) * 3 + (l + 1)];
                }
            }

            // Compute magnitude of gradient
            float grad = sqrt(grad_x * grad_x + grad_y * grad_y);

            //Restrict value to [0, 255] using sigmoid
            grad = 255 * sigmoid(grad);

            //Convert magnitud to integer value
            uint8_t edge = (uint8_t)grad;

            // Set output pixel value to the edge value in all channels;
            // outbuff[i * width + j] = TIFFRGBAImagePackRGBA(edge, edge, edge, 255); // Likely source of error
            outbuff[i * width + j] = (edge << 24) | (edge << 16) | (edge << 8) | (255); 
        }
    }
}

float sigmoid(float x) {
    return (1/(1.0 + exp(-x)));
}