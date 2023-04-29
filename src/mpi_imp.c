/* File:     mpi_imp.c
 *
 * Purpose:  Perform an edge detection convolution of a series of .tif/.tiff images.
 *
 * Compile:  g++ mpi_imp.c -o mpi_imp
 * Run:      ./mpi_imp ./data/<TIFF image(s)>
 *
 * Input:    TIFF image(s) stored in subdirectory ./data.
 * Output:   TIFF image(s) with edge detection convolution applied stored in 
 *           subdirectory ./convoluted
 *
 * Errors:   If an error is detected (no images provided or no images can be opened), the
 *           program prints a message and all processes quit.
 */
#include <iostream>
#include <fstream>
#include <tiffio.h>
#include <vector>
#include <cmath>
#include <string>
#include <cstring>
#include <mpi.h>

void err_check(bool ok, char* err_msg);
void conv_tiff(TIFF* tiff, char* path, const float (&edge_x)[9], const float (&edge_y)[9], int my_rank, int comm_sz);
float sigmoid(float x);

double avgtime = 0;

/*-------------------------------------------------------------------*/
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

    //Initalize MPI
    int my_rank, comm_sz;
    MPI_Init(NULL, NULL);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);

    //Convolute all the tiffs! Also close each tiff after processing it.
    for (int j = 0; j < tiff_points.size(); j++) {
        conv_tiff(tiff_points[j], argv[j + 1], edge_x, edge_y, my_rank, comm_sz);
        TIFFClose(tiff_points[j]);
    };
    
    avgtime = avgtime / tiff_points.size();
    if (my_rank == 0)
        std::cout << "Averge convolution time: " << avgtime << std::endl;

    //Finalize MPI
    MPI_Finalize();
    
    return 0;
}

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
        exit(1);
    }
} /* err_check */

/*-------------------------------------------------------------------
 * Function:  conv_tiff
 * Purpose:   Perform convolution of image using two edge detection 
 *            in x and y directions.
 * In args:   tiff:     pointer to tiff image
 *            kernel_x: used to find gradient along x axis.
 *            kernel_y: used to find gradient along y axis.
 * 
 * Note:      This impementation employs MPI to parallelize the
 *            convolution. It does this by sharing a memory window
 *            among all the processes. Each process then convolutes
 *            part of the image stored in this memory window.
 */
void conv_tiff(
        TIFF* tiff                 /* in  */,
        char* path                 /* in  */, 
        const float (&kernel_x)[9] /* in  */, 
        const float (&kernel_y)[9] /* in  */,
        int my_rank                /* in  */,
        int comm_sz                /* in  */) {

    // Get number of rows and number of columns
    uint32_t width, height; 
    TIFFGetField(tiff, TIFFTAG_IMAGELENGTH, &height);
    TIFFGetField(tiff, TIFFTAG_IMAGEWIDTH, &width);

    // Create a node-local communicator
    MPI_Comm nodecomm; // Declare communicator object (group of processes)
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, &nodecomm); 
    // ^ Create communicator from MPI_COMM_WORLD

    // Create a share memory window for inbuff
    MPI_Win win_in; // Declare window object
    uint32_t* baseptr_in;
    MPI_Win_allocate_shared(height * width * sizeof(uint32_t), sizeof(uint32_t), MPI_INFO_NULL, nodecomm, &baseptr_in, &win_in);
    // ^ Create shared memory window by allocating memory for each process on the same node

    // Create a shared memory window for outbuff
    MPI_Win win_out;
    uint32_t* baseptr_out;
    MPI_Win_allocate_shared(height * width * sizeof(uint32_t), sizeof(uint32_t), MPI_INFO_NULL, nodecomm, &baseptr_out, &win_out);

    // Get a pointer to the shared memory window for inbuff
    int disp_unit_in;
    MPI_Aint size_in; // MPI_Aint -> represetns address integer in MPI
    uint32_t* shmptr_in;
    MPI_Win_shared_query(win_in, 0, &size_in, &disp_unit_in, &shmptr_in);
    // ^ Queries process-local address for remote memory segment created with MPI_Win_allocate_shared

    // Get a pointer to the shared memory window for outbuff
    int disp_unit_out;
    MPI_Aint size_out;
    uint32_t* shmptr_out;
    MPI_Win_shared_query(win_out, 0, &size_out, &disp_unit_out, &shmptr_out);

    if (my_rank == 0)
        TIFFReadRGBAImage(tiff, width, height, shmptr_in, 0);
    avgtime -= MPI_Wtime();
    MPI_Barrier(nodecomm);

    int my_numrows = height/comm_sz; // Number of rows assigned to this process
    int my_startrow = my_numrows * my_rank;
    int rowsleft = height % comm_sz;
    if ((my_rank == (comm_sz - 1)) & (rowsleft != 0))
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
                    x = std::max(0, std::min(x, (int)width - 1));
                    y = std::max(0, std::min(y, (int)height - 1));

                    // Get pixel value in the input image
                    uint32_t pixel = shmptr_in[y * width + x];

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
            shmptr_out[i * width + j] = (edge << 24) | (edge << 16) | (edge << 8) | (255); 
        }
    }

    MPI_Barrier(nodecomm);
    avgtime += MPI_Wtime();

    if (my_rank == 0) {
        std::string fname_str = "_conv.tiff";
        std::string fnameorig (path);
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

            for (int i = 0; i < height; i ++) {
                // Write one scanline at a time
                TIFFWriteScanline(outimg, &shmptr_out[(height - i - 1) * width], i);
            }
            TIFFClose(outimg);
        }
        std::cout << "Convoluted file " << fnameorig.substr(1) << ".tiff" << std::endl;
    }

    MPI_Barrier(nodecomm);
    MPI_Win_free(&win_in);
    MPI_Win_free(&win_out);
    
} /* conv_tiff */

/*-------------------------------------------------------------------
 * Function:  sigmoid
 * Purpose:   Given an input x, evaluates the sigmoid function
 * In args:   x: value at which the sigmoid function is to be evaluated at
 */
float sigmoid(
        float x /* in */) {
    return (1/(1.0 + exp(-x)));
} /* sigmoid */
