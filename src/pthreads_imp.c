#include <iostream>
#include <fstream>
#include <tiffio.h>
#include <vector>
#include <cmath>
#include <string>
#include <cstring>
#include <pthread.h>

void conv_tiff(uint32_t *inbuff, uint32_t *outbuff, int width, int height, const float (&kernel_x)[9], const float (&kernel_y)[9], int my_startrow, int my_endrow);
float sigmoid(float x);
void *thread_proc(void *args);

struct thread_args {
    uint32_t *inbuff;
    uint32_t *outbuff;
    uint32_t width;
    uint32_t height;

    int thread;
    long thread_count;

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

    int done_creating_flag = 0;
    int done_conv_flag = 0;
};

int main(int argc, char* argv[]) {

    // Declare neccesary variables for pthreads.
    pthread_t* thread_handles;
    struct thread_args ta;

    // Get number of threads from command line
    ta.thread_count = strtol(argv[1], NULL, 10);

    // Get thread handles
    thread_handles = (pthread_t *) malloc(ta.thread_count * sizeof(pthread_t));

    // Initialize vector TIFF pointers
    std::vector<TIFF*> tiff_points;
    std::ifstream file_point;
    for (int i = 2; i < argc; i++) {
        file_point.open(argv[i]);
        if (file_point) {
            tiff_points.push_back(TIFFOpen((argv[i]), "r"));
        } else {
            std::cout << "WARNING: \'" << argv[i] << "\' could not be opened." << std::endl;
        }
        file_point.close();
    };

    // Convolute all the tiffs! Also close each tiff after processing it.
    for (int j = 0; j < tiff_points.size(); j++) {

        TIFFGetField(tiff_points[j], TIFFTAG_IMAGELENGTH, &(ta.height));
        TIFFGetField(tiff_points[j], TIFFTAG_IMAGEWIDTH, &(ta.width));

        // Make space in memory for input buffer and output buffer.
        ta.inbuff = (uint32_t*) _TIFFmalloc(ta.height * ta.width * sizeof(uint32_t));
        ta.outbuff = (uint32_t*) _TIFFmalloc(ta.height * ta.width * sizeof(uint32_t));

        // Read the image data from tiff and store it in inbuff
        TIFFReadRGBAImage(tiff_points[j], ta.width, ta.height, ta.inbuff, 0);

        // std::cout << ta.height << std::endl;
        // std::cout << ta.width << std::endl;
        // std::cout << ta.inbuff << std::endl;

        // Perform convolution
        ta.done_conv_flag = 0;
        for (int thread = 0; thread < ta.thread_count; thread++) {
            std::cout << "Starting thread #" << thread << std::endl;
            ta.thread = thread;
            std::cout << "ta.thread is now " << ta.thread << std::endl;
            pthread_create(&thread_handles[thread], NULL, thread_proc, (void *) &ta);
            while (ta.done_creating_flag == 0);
            ta.done_creating_flag = 0;
        }

        if (ta.thread == ta.thread_count - 1) {
            ta.done_conv_flag = 1;
        }
        while(ta.done_conv_flag == 0);

        for (int thread = 0; thread < ta.thread_count; thread++) {
            pthread_join(thread_handles[thread], NULL);
        }

        std::cout << "Threads joined" << std::endl;
        // while(ta.done_conv_flag == 0);
        TIFFClose(tiff_points[j]);
        
        std::cout << "File name created" << std::endl;

        std::string fname_str = "./pconv.tif";
        // std::string fnameorig (argv[j+1]);
        // std::string sub_fnameorig = fnameorig.substr(fnameorig.length() - 12);
        // fname_str += sub_fnameorig;

        char fname[100];
        std::strcpy(fname, fname_str.c_str());


        // Construct tiff from outbuff.
        TIFF* outimg = TIFFOpen(fname, "w");
        if (outimg) {
            std::cout << "Started creating TIFF" << std::endl;
            TIFFSetField(outimg, TIFFTAG_IMAGEWIDTH, ta.width);
            TIFFSetField(outimg, TIFFTAG_IMAGELENGTH, ta.height);
            TIFFSetField(outimg, TIFFTAG_BITSPERSAMPLE, 8);
            TIFFSetField(outimg, TIFFTAG_SAMPLESPERPIXEL, 4);
            TIFFSetField(outimg, TIFFTAG_PLANARCONFIG, PLANARCONFIG_CONTIG);
            TIFFSetField(outimg, TIFFTAG_PHOTOMETRIC, PHOTOMETRIC_RGB);

            for (int i = 0; i < ta.height; i ++) {
                // Write one scanline at a time
                TIFFWriteScanline(outimg, &ta.outbuff[(ta.height - i - 1) * ta.width], i);
            }

            TIFFClose(outimg);
        }

        delete[] ta.inbuff;
        delete[] ta.outbuff;

        std::cout << "Yay!" << std::endl;
    };
    
    return 0;
}

void *thread_proc (void *args) {    
    struct thread_args *ta = (struct thread_args *) args;
    std::cout << ta->thread <<" got to thread_proc!" << std::endl;

    int my_numrows = (ta->height)/(ta->thread_count); // Number of rows assigned to this process
    int my_startrow = my_numrows * ta->thread;
    int rowsleft = ta->height % ta->thread_count;
    if ((ta->thread == (ta->thread_count - 1)) & (rowsleft != 0)) {
        my_numrows += rowsleft;
    }
    int my_endrow = my_startrow + my_numrows;
    // std::cout << "thread 1 start: " << my_startrow << "; end:" << my_endrow << std::endl;
    ta->done_creating_flag = 1;

    conv_tiff(ta->inbuff, ta->outbuff, ta->width, ta->height, ta->edge_x, ta->edge_y, my_startrow, my_endrow);
}

void conv_tiff(uint32_t *inbuff, uint32_t *outbuff, int width, int height, const float (&kernel_x)[9], const float (&kernel_y)[9], int my_startrow, int my_endrow) {
    std::cout << "One thread got to conv_tiff" << std::endl;
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
                    x = std::max(0, std::min(x, (int)width - 1));
                    y = std::max(0, std::min(y, (int)height - 1));

                    // Get pixel value in the input image
                    // std::cout << "One thread is checking inbuff" << std::endl;
                    uint32_t pixel = inbuff[y * width + x];
                    // std::cout << "One thread finished checking inbuff" << std::endl;

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

            // std::cout << "One pixel calculated. Storing..." << std::endl;
            // Set output pixel value to the edge value in all channels;
            outbuff[i * width + j] = (edge << 24) | (edge << 16) | (edge << 8) | (255);
            // std::cout << "Pixel stored!" << std::endl; 
        }
    }
}

float sigmoid(float x) {
    return (1/(1.0 + exp(-x)));
}