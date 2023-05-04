/* File:     cuda_imp.c
 *
 * Purpose:  Perform an edge detection convolution of a series of .tif/.tiff images.
 *
 * Compile:  nvcc cuda_imp.cu -o cuda_imp -ltiff
 * Run:      ./cuda_imp <Number of threads> <Number of blocks> ./data/<TIFF image(s)>
 *
 * Input:    TIFF image(s) stored in subdirectory ./data.
 * Output:   TIFF image(s) with edge detection convolution applied stored in 
 *           subdirectory ./convoluted
 *
 * Errors:   If an error is detected (no images provided or no images can be opened), the
 *           program prints a message and quits.
 *
 */
#include <iostream>
#include <fstream>
#include <tiffio.h>
#include <vector>
#include <cmath>
#include <string>
#include <cstring>
#include <cuda.h>
#include <chrono>

__global__ void conv_tiff(uint32_t* inbuff, uint32_t* outbuff, int width, int height);

float time_avg = 0;

/*-------------------------------------------------------------------*/
int main(int argc, char* argv[]) {

    // Get number of threads and blocks from command line
    int blk_ct = std::stoi(argv[1], NULL, 10);
    int th_per_blk = std::stoi(argv[2], NULL, 10);

    // Initialize vector TIFF pointers
    std::vector<TIFF*> tiff_points;
    std::ifstream file_point;
    for (int i = 3; i < argc; i++) {
        file_point.open(argv[i]);
        if (file_point)
            tiff_points.push_back(TIFFOpen((argv[i]), "r"));
        else
            std::cout << "WARNING: \'" << argv[i] << "\' could not be opened." << std::endl;
        file_point.close();
    };
    
    // Convolute all the tiffs! Also close each tiff after processing it.
    for (int j = 0; j < tiff_points.size(); j++) {

        int width, height;
        uint32_t* inbuff;
        uint32_t* outbuff;

        // Read image parameters from tiff
        TIFFGetField(tiff_points[j], TIFFTAG_IMAGELENGTH, &height);   
        TIFFGetField(tiff_points[j], TIFFTAG_IMAGEWIDTH, &width);
        
        // Allocate memory in host and GPU
        cudaMallocManaged(&inbuff, height * width * sizeof(uint32_t));
        cudaMallocManaged(&outbuff, height * width * sizeof(uint32_t));

        // Read image data from tiff
        TIFFReadRGBAImage(tiff_points[j], width, height, inbuff, 0);

        auto start = std::chrono::high_resolution_clock::now();
        // Start convolution in GPU
        conv_tiff<<<blk_ct, th_per_blk>>>(inbuff, outbuff, width, height);
        cudaDeviceSynchronize();
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
        time_avg += static_cast<float>(duration.count());
        TIFFClose(tiff_points[j]);
        
        std::string fname_str = "_conv.tiff";
        std::string fnameorig (argv[j + 3]);
        fnameorig = (fnameorig.substr(fnameorig.size() - 13, 13)).substr(0, 9);
        fname_str = "./convoluted/" + fnameorig + fname_str;
        char fname[100];
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

            for (int i = 0; i < height; i ++) // Write one scanline at a time
                TIFFWriteScanline(outimg, &outbuff[(height - i - 1) * width], i);
            TIFFClose(outimg);
        }

        std::cout << "Convoluted file " << fnameorig.substr(1) << ".tiff" << std::endl;

        // Free memory
        cudaFree(inbuff);
        cudaFree(outbuff);

    };
    
    time_avg = time_avg / tiff_points.size();
    std::cout << "Average convolution time: " << time_avg * pow(10,-9) << " seconds" << std::endl;

    return 0;
} /* main */

/*-------------------------------------------------------------------
 * Function:  conv_tiff
 * Purpose:   Perform convolution of image using two edge detection 
 *            in x and y directions.
 * In args:   ta:          struct holding all data the threads use
 *                         during convolution
 *            my_startrow: index from which the thread will start
 *                         convolution
 *            my_endrow:   index on which the thread will stop
 *                         convolution
 */
__global__ void conv_tiff(
        uint32_t* inbuff,  /* in  */
        uint32_t* outbuff, /* in  */
        int width,         /* in  */
        int height         /* in  */ ) {    

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

    int my_index = blockIdx.x * blockDim.x + threadIdx.x;
    int thread_count = gridDim.x * blockDim.x;
    
    int my_numrows = height / thread_count;
    int my_startrow = my_numrows * my_index;
    int rowsleft = height % thread_count;
    if ((my_index == (thread_count - 1)) & (rowsleft != 0))
        my_numrows += rowsleft;
    int my_endrow = my_startrow + my_numrows;

    for (int i = my_startrow; i < my_endrow; i++) {
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
                    x = max(0, min(x, (int)width - 1));
                    y = max(0, min(y, (int)height - 1));

                    // Get pixel value in the input image
                    uint32_t pixel = inbuff[y * width + x];

                    uint8_t r = TIFFGetR(pixel);
                    uint8_t g = TIFFGetG(pixel);
                    uint8_t b = TIFFGetB(pixel);

                    // Convert pixel value to grayscale
                    float gray = 0.2126f * r + 0.7152f * g + 0.0722f * b;

                    // Multiply pixel value with corresponding kernel element and add to the gradient values
                    grad_x += gray * edge_x[(k + 1) * 3 + (l + 1)];
                    grad_y += gray * edge_y[(k + 1) * 3 + (l + 1)];
                }
            }

            // Compute magnitude of gradient
            float grad = sqrt(grad_x * grad_x + grad_y * grad_y);

            //Restrict value to [0, 255] using sigmoid
            grad = 255 * 1/(1.0 + exp(-1 * grad));
            
            //Convert magnitud to integer value
            uint8_t edge = (uint8_t)grad;

            // Set output pixel value to the edge value in all channels;
            outbuff[i * width + j] = (edge << 24) | (edge << 16) | (edge << 8) | (255);
        }
    }
} /* conv_tiff */

