#pragma once

// maximum magnitude of an element of the mandelbrot set, squared
#define MAX_MAG_SQ 4

// use first device found
#define DEVICE 0

/*
 * All CUDA calls and fractal generation handled here
 */
class cudaGraphicsResource;
class CudaMandelbrot
{
private:
    int max_block_size_x;
    int max_block_size_y;

    int max_grid_size_x;
    int max_grid_size_y;

    int image_width;
    int image_height;

    int iterations;

    unsigned int* h_dwell_map;
    float* h_image_colours;

    unsigned int* d_dwell_map;
    float* d_image_colours;

    cudaGraphicsResource ** pbo_resource;

public:
    CudaMandelbrot(int image_width, int image_height, int iter, unsigned int pbo);
    ~CudaMandelbrot();

    void runCUDA();

    inline
    void setImageWidth(int width) { image_width = width; }
    
    inline
    void setImageHieght(int height) { image_height = height; }

    inline
    void setIterations(int i) { if (iterations > 0) iterations = i; }
};
