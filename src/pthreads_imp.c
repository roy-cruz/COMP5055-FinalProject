/* File:     pthreads_imp.c
 *
 * Purpose:  Perform an edge detection convolution of a series of .tif/.tiff images.
 *
 * Compile:  g++ pthreads_imp.c -o pthreads_imp -ltiff -pthreads
 * Run:      ./pthreads_imp <Number of threads> ./data/<TIFF image(s)>
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
#include <pthread.h>
#include <chrono>

// void err_check(bool ok, char* err_msg);
void err_check(bool ok, char* err_msg);
void conv_tiff(struct thread_args *ta, int my_startrow, int my_endrow);
float sigmoid(float x);
void *thread_proc(void *args);

struct thread_args {
    uint32_t *inbuff;
    uint32_t *outbuff;
    uint32_t width;
    uint32_t height;

    TIFF* tiff_point;

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
    int done_conv_num = 0;

    std::chrono::time_point<std::chrono::high_resolution_clock> start, end;
    std::chrono::nanoseconds duration;
    float time_avg;
};

/*-------------------------------------------------------------------*/
int main(int argc, char* argv[]) {
    err_check((argc >= 3) & (atoi(argv[1]) != 0), "ERROR: Not enough command line inputs or threads.");

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

    // Check that at least one tiff was opened.
    err_check(tiff_points.size() != 0, "ERROR: No tiff file could be opened.");

    // Convolute all the tiffs! Also close each tiff after processing it.
    for (int j = 0; j < tiff_points.size(); j++) {
        // Perform convolution
        ta.done_conv_num = 0;
        ta.tiff_point = tiff_points[j];
        for (int thread = 0; thread < ta.thread_count; thread++) {
            // std::cout << "Starting thread #" << thread << std::endl;
            ta.thread = thread;
            // std::cout << "ta.thread is now " << ta.thread << std::endl;
            pthread_create(&thread_handles[thread], NULL, thread_proc, (void *) &ta);
            while (ta.done_creating_flag == 0);
            ta.done_creating_flag = 0;
        }

        while(ta.done_conv_num != ta.thread_count);

        for (int thread = 0; thread < ta.thread_count; thread++)
            pthread_join(thread_handles[thread], NULL);

        TIFFClose(tiff_points[j]);
        
        std::string fname_str = "_conv.tiff";
        std::string fnameorig (argv[j + 2]);
        fnameorig = (fnameorig.substr(fnameorig.size() - 13, 13)).substr(0, 9);
        fname_str = "./convoluted/" + fnameorig + fname_str;
        char fname[100];
        std::strcpy(fname, fname_str.c_str());

        // Construct tiff from outbuff.
        TIFF* outimg = TIFFOpen(fname, "w");
        if (outimg) {
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

        std::cout << "Convoluted file " << fnameorig.substr(1) << ".tiff" << std::endl;

        delete[] ta.inbuff;
        delete[] ta.outbuff;

    };
    
    ta.time_avg = ta.time_avg / tiff_points.size();
    std::cout << "Average convolution time: " << ta.time_avg * pow(10,-9) << " seconds" << std::endl;

    return 0;
} /* main */

/*-------------------------------------------------------------------
 * Function:  err_check
 * Purpose:   Check whether there is an error. Terminate program if so.
 * In args:   ok:      false if calling process has found an error, true
 *                     otherwise
 *            err_msg: error message to be printed
 */
void err_check(
        bool ok       /* in  */, 
        char* err_msg /* in  */){
    if (ok == false) {
        std::cerr << err_msg << std::endl;
        exit(-1);
    }
} /* err_check */

/*-------------------------------------------------------------------
 * Function:  thread_proc
 * Purpose:   Initializes all the variables the thread will use
 *            in the convolution. Also allocated memory neccessary
 *            for input and output buffer.
 * In args:   args:   void variable which is used to pass the struct 
 *                    'ta' which holds all the data used by the 
 *                    threads
 */
void *thread_proc(
        void *args /* in  */) {    
    struct thread_args *ta = (struct thread_args *) args;
    
    if (ta->thread == 0) {
        TIFFGetField(ta->tiff_point, TIFFTAG_IMAGELENGTH, &(ta->height));
        TIFFGetField(ta->tiff_point, TIFFTAG_IMAGEWIDTH, &(ta->width));

        // Make space in memory for input buffer and output buffer.
        ta->inbuff = (uint32_t*) _TIFFmalloc(ta->height * ta->width * sizeof(uint32_t));
        ta->outbuff = (uint32_t*) _TIFFmalloc(ta->height * ta->width * sizeof(uint32_t));

        // Read the image data from tiff and store it in inbuff
        TIFFReadRGBAImage(ta->tiff_point, ta->width, ta->height, ta->inbuff, 0);
    }

    int my_numrows = (ta->height)/(ta->thread_count); // Number of rows assigned to this process
    int my_startrow = my_numrows * ta->thread;
    int rowsleft = ta->height % ta->thread_count;
    if ((ta->thread == (ta->thread_count - 1)) & (rowsleft != 0)) {
        my_numrows += rowsleft;
    }
    int my_endrow = my_startrow + my_numrows;

    if(ta->thread == 0)
        ta->start = std::chrono::high_resolution_clock::now();
    ta->done_creating_flag = 1;

    conv_tiff(ta, my_startrow, my_endrow);

    if(ta->done_conv_num + 1 == ta->thread_count) {
        ta->end = std::chrono::high_resolution_clock::now();
        ta->duration = std::chrono::duration_cast<std::chrono::nanoseconds>(ta->end - ta->start);
        ta->time_avg += static_cast<float>(ta->duration.count());
    }
    ta->done_conv_num += 1;
} /* thread_proc */

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
void conv_tiff(
        struct thread_args *ta, /* in  */
        int my_startrow,        /* in  */
        int my_endrow           /* in  */) {    

    for (int i = my_startrow; i < my_endrow; i++) {
        for (int j = 0; j < ta->width; j++) {
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
                    x = std::max(0, std::min(x, (int)ta->width - 1));
                    y = std::max(0, std::min(y, (int)ta->height - 1));

                    // Get pixel value in the input image
                    uint32_t pixel = ta->inbuff[y * ta->width + x];

                    uint8_t r = TIFFGetR(pixel);
                    uint8_t g = TIFFGetG(pixel);
                    uint8_t b = TIFFGetB(pixel);

                    // Compute luminance
                    float gray = 0.2126f * r + 0.7152f * g + 0.0722f * b;

                    // Multiply pixel value with corresponding kernel element and add to the gradient values
                    grad_x += gray * ta->edge_x[(k + 1) * 3 + (l + 1)];
                    grad_y += gray * ta->edge_y[(k + 1) * 3 + (l + 1)];
                }
            }

            // Compute magnitude of gradient
            float grad = sqrt(grad_x * grad_x + grad_y * grad_y);

            //Restrict value to [0, 255] using sigmoid
            grad = 255 * sigmoid(grad);

            //Convert magnitud to integer value
            uint8_t edge = (uint8_t)grad;

            // Set output pixel value to the edge value in all channels;
            ta->outbuff[i * ta->width + j] = (edge << 24) | (edge << 16) | (edge << 8) | (255);
        }
    }
} /* conv_tiff */

/*-------------------------------------------------------------------
 * Function:  sigmoid
 * Purpose:   Given an input x, evaluates the sigmoid function
 * In args:   x: value at which the sigmoid function is to be evaluated at
 */
float sigmoid(float x) {
    return (1/(1.0 + exp(-x)));
} /* sigmoid */