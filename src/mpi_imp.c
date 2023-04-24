#include <iostream>
#include <fstream>
#include <tiffio.h>
#include <vector>
#include <cmath>
#include <string>
#include <cstring>
#include <mpi.h>

void err_check(bool is_err, char* err_msg);
void conv_tiff(TIFF* tiff, const float (&edge_x)[9], const float (&edge_y)[9]);
float sigmoid(float x);

int numprocd;

int main(int argc, char* argv[]) {
    bool is_err = false; 

    // Check that at least one argument is given.
    is_err = argc < 2;
    err_check(is_err, "ERROR: No imput files.");

    // Initialize vector TIFF pointers
    std::vector<TIFF*> tiff_points;
    std::ifstream file_point;
    for (int i = 1; i < argc; i++) {
        file_point.open(argv[i]);
        if (file_point) {
            tiff_points.push_back(TIFFOpen((argv[i]), "r"));
        } else {
            std::cout << "WARNING: \'" << argv[i] << "\' could not be opened." << std::endl;
        }
        file_point.close();
    };

    // Check that at least one tiff was opened.
    is_err = (tiff_points.size() == 0);
    err_check(is_err, "ERROR: No tiff file could be opened.");

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

    numprocd = 0;

    //Convolute all the tiffs! Also close each tiff after processing it.
    for (int j = 0; j < tiff_points.size(); j++) {
        conv_tiff(tiff_points[j], edge_x, edge_y);
        TIFFClose(tiff_points[j]);
        numprocd++;
    };
    
    return 0;
}

void err_check(bool is_err, char* err_msg){
    if (is_err == true) {
        std::cerr << err_msg << std::endl;
        exit(1);
    };    
};

void conv_tiff(TIFF* tiff, const float (&kernel_x)[9], const float (&kernel_y)[9]) {
    // Get number of rows and number of columns
    uint32_t width, height; 
    TIFFGetField(tiff, TIFFTAG_IMAGELENGTH, &height);
    TIFFGetField(tiff, TIFFTAG_IMAGEWIDTH, &width);

    // // Make space in memory for input buffer and output buffer.
    // uint32_t* inbuff = (uint32_t*) _TIFFmalloc(height * width * sizeof(uint32_t));
    // uint32_t* outbuff = (uint32_t*) _TIFFmalloc(height * width * sizeof(uint32_t));

    // // Read the image data from tiff and store it in inbuff
    // TIFFReadRGBAImage(tiff, width, height, inbuff, 0);

    int my_rank, comm_sz;
    MPI_Init(NULL, NULL);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);

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

    if (my_rank == 0) {
        TIFFReadRGBAImage(tiff, width, height, shmptr_in, 0);
    }

    MPI_Barrier(nodecomm);

    int my_numrows = height/comm_sz; // Number of rows assigned to this process
    int my_startrow = my_numrows * my_rank;
    int rowsleft = height % comm_sz;
    if ((my_rank == (comm_sz - 1)) & (rowsleft != 0)) {
        my_numrows += rowsleft;
    }
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
            // outbuff[i * width + j] = TIFFRGBAImagePackRGBA(edge, edge, edge, 255); // Likely source of error
            shmptr_out[i * width + j] = (edge << 24) | (edge << 16) | (edge << 8) | (255); 
        }
    }

    MPI_Barrier(nodecomm);

    if (my_rank == 0) {
        std::string fname_str = "./conv_img_";
        fname_str += std::to_string(numprocd);
        fname_str += "_mpi.tiff";

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
    }

    MPI_Barrier(nodecomm);
    MPI_Win_free(&win_in);
    MPI_Win_free(&win_out);
    MPI_Finalize();
    
    std::cout << "Yay!" << std::endl;
}

float sigmoid(float x) {
    return (1/(1.0 + exp(-x)));
}
