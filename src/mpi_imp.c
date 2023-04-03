#include <iostream>
#include <fstream>
#include <tiffio.h>
#include <vector>
#include <array>
#include <cstdlib>
#include <cmath>
#include <mpi.h>

void err_check(bool is_err, char* err_msg);
void conv_tiff(TIFF* img, const float (&kernel)[9]);
float sigmoid(float x) {
    return (255.0/(1.0 + exp(-x)));
};


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

    const float kernel[9] = {
        -1.0, -1.0, -1.0,
        -1.0, 8.0, -1.0,
        -1.0, -1.0, -1.0
    };

    for (int j = 0; j < fnames.size(); j++) {
        TIFF* img = TIFFOpen(fnames[j].c_str(), "r");
        conv_tiff(img, kernel);
        TIFFClose(img);
    }
    
    return 0;
}

void err_check(bool is_err, char* err_msg){
    if (is_err == true) {
        std::cerr << err_msg << std::endl;
        exit(1);
    };    
};

void conv_tiff(TIFF* tiff, const float (&kernel)[9]) {
    uint32_t columns, rows; // columns -> number of columns, rows -> number of rows
    int rank, size;

    MPI_Init(NULL, NULL);
    MPI_Comm_size(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    TIFFGetField(tiff, TIFFTAG_IMAGELENGTH, &rows);
    TIFFGetField(tiff, TIFFTAG_IMAGEWIDTH, &columns);

    if (rank == 0) {
        uint32_t* inbuff = (uint32_t*) _TIFFmalloc(rows * columns * sizeof(uint32_t));
        uint32_t* outbuff = (uint32_t*) _TIFFmalloc(rows * columns * sizeof(uint32_t));

        TIFFReadRGBAImage(tiff, columns, rows, inbuff);
    }

    MPI_Barrier(MPI_COMM_WORLD);

    uint32_t chunck_size = rows / size;
    uint32_t remainder = rows % size;
    uint32_t my_start_row, my_num_rows;
    if (rank < remainder) {
        my_num_rows = chunck_size + 1;
        my_start_row = rank * my_num_rows;
    } else {
        my_num_rows = chunck_size;
        my_start_row = rank * my_num_rows + remainder;
    }

    // Arrays for components
    std::vector<uint8_t> r(my_num_rows * columns);
    std::vector<uint8_t> g(my_num_rows * columns);
    std::vector<uint8_t> b(my_num_rows * columns);

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
        TIFFSetField(outimg, TIFFTAG_PLANARCONFIG, 2);
        TIFFSetField(outimg, TIFFTAG_PHOTOMETRIC, 2);
        TIFFWriteEncodedStrip(outimg, 0, outbuff, columns * rows * sizeof(uint32_t));
        TIFFClose(outimg);
    }

    delete[] inbuff;
    delete[] outbuff;

    std::cout << "Yay!" << std::endl;


}

    // Initialize MPI

    // Extract the image data from the tiff and store in an uint32_t array (inbuff)

    // Share the address to the uint32_t array to all cores. Also share the address of the output array (outbuff)

    // Calculate what section of the data each core is responsible for.

    // Each core does a convolution on its section and stores the result in outbuff
    
    // Load each file and send them to be convoluted
























// #include <iostream>
// #include <fstream>
// #include <tiffio.h>
// #include <vector>
// #include <array>
// #include <cstdlib>
// #include <cmath>
// #include <mpi.h>

// void err_check(bool is_err, char* err_msg);
// void start_process(std::vector<std::string> fnames);
// void conv_tiff(TIFF* img, const float (&kernel)[9]);
// float sigmoid(float x);


// int main(int argc, char* argv[]) {
//     bool is_err = false; 

//     // Check that at least one argument is given.
//     if (argc < 2) {
//         is_err = true;
//     }
//     err_check(is_err, "ERROR: No imput files.");

//     // Initialize vector with filenames.
//     std::vector<std::string> fnames;
//     std::ifstream file_point;
//     for (int i = 1; i < argc; i++) {
//         file_point.open(argv[i]);
//         if (file_point) {
//             fnames.push_back(std::string(argv[i]));
//         }
//         file_point.close();
//     }

//     start_process(fnames);
    
//     return 0;
// }

// void err_check(bool is_err, char* err_msg){
//     if (is_err == true) {
//         std::cerr << err_msg << std::endl;
//         exit(1);
//     };    
// };

// void start_process(std::vector<std::string> fnames){

//     // Initialize kernel
//     const float kernel[9] = {
//         -1.0, -1.0, -1.0,
//         -1.0, 8.0, -1.0,
//         -1.0, -1.0, -1.0
//     };

//     // Load each file and send them to be convoluted
//     for (int j = 0; j < fnames.size(); j++) {
//         if (rank == 0) {
//             TIFF* img = TIFFOpen(fnames[j].c_str(), "r");
//         }
//         conv_tiff(img, kernel);
//     }
// };

// void conv_tiff(TIFF* tiff, const float (&kernel)[9]) {
//     uint32_t columns, rows; // columns -> number of columns, rows -> number of rows
//     int rank, size, bits_per_sample, samples_per_pixel;

//     //Initialize MPI
//     MPI_Init(NULL, NULL);
//     MPI_Comm_rank(MPI_COMM_WORLD, &rank);
//     MPI_Comm_size(MPI_COMM_WORLD, &size);

//     TIFFGetField(tiff, TIFFTAG_IMAGELENGTH, &rows);
//     TIFFGetField(tiff, TIFFTAG_IMAGEWIDTH, &columns);

//     uint32_t chunck_size = rows / size;
//     uint32_t remainder = rows % size;
//     uint32_t my_start_row, my_num_rows;
//     if (rank < remainder) {
//         my_num_rows = chunck_size + 1;
//         my_start_row = rank * my_num_rows;
//     } else {
//         my_num_rows = chunck_size;
//         my_start_row = rank * my_num_rows + remainder;
//     }

//     // Each core allocates enough memory to use for itself
//     uint32_t row_size = columns * sizeof(uint32_t);
//     uint32_t buff_size = my_num_rows * row_size;
//     uint32_t* inbuff = (uint32_t*) _TIFFmalloc(buff_size);
//     uint32_t* outbuff = (uint32_t*) _TIFFmalloc(buff_size); 

//     // Each core stores the part of the image that corresponds to it
//     uint32_t current_row = my_start_row;
//     for (uint32_t i = 0; i < my_num_rows; i++) {
//         TIFFReadScanline(tiff, inbuff + i * columns, current_row);
//         current_row++;
//     }

//     TIFFClose(tiff);






//     if (rank == 0) {
//         uint32_t* inbuff = (uint32_t*) _TIFFmalloc(rows * columns * sizeof(uint32_t));
//         uint32_t* outbuff = (uint32_t*) _TIFFmalloc(rows * columns * sizeof(uint32_t));
        
//         // Arrays for components
//         std::vector<uint8_t> r(rows * columns);
//         std::vector<uint8_t> g(rows * columns);
//         std::vector<uint8_t> b(rows * columns);

        
//     }


    



//     for (int i = 0; i < rows * columns; i++) {
//         r[i] = TIFFGetR(inbuff[i]);
//         g[i] = TIFFGetG(inbuff[i]);
//         b[i] = TIFFGetB(inbuff[i]);
//     }

//     float sum_r, sum_g, sum_b, sum_a;
//     int radius = 1, kernel_size = 3; // Radius of convolution kernel and size of kernel
//     int pixel_x, pixel_y, pixel_index;
//     float weight;
//     uint32_t conv_pixel;

//     // Convolution
//     for (int j = 0; j < rows; j++) {
//         for (int i = 0; i < columns; i++) {
//             sum_r = 0; sum_g = 0.0; sum_b = 0.0;
//             for (int y = -radius; y <= radius; y++) {
//                 for (int x = -radius; x <= radius; x++) {
//                     pixel_y = j + y;
//                     pixel_x = i + x;
//                     if (pixel_x >= 0 && pixel_x < columns && pixel_y >= 0 && pixel_y < rows) { // Check that its not over the edge
//                         pixel_index = pixel_y * columns + pixel_x;
//                         weight = kernel[(y + radius) * kernel_size + (x + radius)];
//                         sum_r += r[pixel_index] * weight;
//                         sum_g += g[pixel_index] * weight;
//                         sum_b += b[pixel_index] * weight;
//                     }
//                 }
//             }

//             int sum_rc = uint32_t(sigmoid(sum_r));
//             int sum_gc = uint32_t(sigmoid(sum_g));
//             int sum_bc = uint32_t(sigmoid(sum_b));

//             // At this point, we have calculated one of the convoluted picture's pixel. We can proceed to store the calculated values.
//             conv_pixel = (sum_rc << 16) | (sum_gc << 8) | sum_bc;

//             // Store result in outbuff
//             outbuff[j * columns + i] = conv_pixel;

//         }
//     }

//     TIFF* outimg = TIFFOpen("conv_img.tif", "w");
//     if (outimg) {
//         TIFFSetField(outimg, TIFFTAG_IMAGEWIDTH, columns);
//         TIFFSetField(outimg, TIFFTAG_IMAGELENGTH, rows);
//         TIFFSetField(outimg, TIFFTAG_SAMPLESPERPIXEL, 3);
//         TIFFSetField(outimg, TIFFTAG_BITSPERSAMPLE, 16);
//         // TIFFSetField(outimg, TIFFTAG_ORIENTATION, ORIENTATION_TOPLEFT);
//         TIFFSetField(outimg, TIFFTAG_PLANARCONFIG, 2);
//         TIFFSetField(outimg, TIFFTAG_PHOTOMETRIC, 2);
//         TIFFWriteEncodedStrip(outimg, 0, outbuff, columns * rows * sizeof(uint32_t));
//         TIFFClose(outimg);
//     }

//     delete[] inbuff;
//     delete[] outbuff;

//     std::cout << "Yay!" << std::endl;
// }

// float sigmoid(float x) {
//     return (255.0/(1.0 + exp(-x)));
// }