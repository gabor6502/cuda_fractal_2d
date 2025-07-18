#include "kernel.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdlib.h>
#include <stdio.h>
#include <complex>

struct ImageParams
{
    int width;
    int height;
};
ImageParams imageParams;

struct HostBuffers
{
    unsigned int * h_dwell_map;
    float * h_image_colours;
};
HostBuffers hostBuffers;

struct DeviceBuffers
{
    unsigned int * d_dwell_map;
    float * d_image_colours;
};
DeviceBuffers deviceBuffers;

typedef std::complex<double> complex;

// maximum magnitude of an element of the mandelbrot set, squared
#define MAX_MAG_SQ 4

// the standard cuda return checking macro
#define CUDA_CHECK_RETURN(value)                                     \
{                                                                    \
    cudaError_t _m_cudaStat = value;                                 \
    if ( _m_cudaStat != cudaSuccess)                                 \
    {                                                                \
        fprintf(stderr, "Error %s at line %d in file %s\n",          \
            cudaGetErrorString(_m_cudaStat), __LINE__, __FILE__);    \
        exit(EXIT_FAILURE);                                          \
    }                                                                \
} 


// starting off with basic alg, no DP/mariani-silver yet

__global__ void k_mandelbrot(int * d_dwell_map, int max_iterations,
                                   int width_dwell_map, int height_dwell_map, 
                                   complex bottom_left, complex top_right, 
                                   int pixel_x, int pixel_y)
{
    int img_x = threadIdx.x + blockDim.x * blockIdx.x;
    int img_y = threadIdx.y + blockDim.y * blockIdx.y;

    if (img_x < width_dwell_map && img_y < height_dwell_map)
    {
        d_dwell_map[img_y * height_dwell_map + img_x] 
            = mandelbrot_pixel(max_iterations, width_dwell_map, height_dwell_map, 
                               bottom_left, top_right, pixel_x, pixel_y);
    }

}

__device__ int mandelbrot_pixel(int max_iterations,
                                int width_dwell_map, int height_dwell_map,
                                complex bottom_left, complex top_right,
                                int pixel_x, int pixel_y)
{
    // convert from image-segment space to mandelbrot space
    complex dist_max_min = top_right - bottom_left;
    complex c = bottom_left + complex((float)pixel_x / (float)width_dwell_map * dist_max_min.real(),
                                      (float)pixel_y / (float)height_dwell_map * dist_max_min.imag());
    complex z = c; // iterate starting at c

    int dwell = 0;
    while(dwell++ < max_iterations && z.real()*z.real() + z.imag()*z.imag() < MAX_MAG_SQ)
    {
        z = z * z + c;
    }

    return dwell;
}

// assign colours of the screen from dwell-map

__global__ void k_colour_dwell_map(float * d_pixels, int * d_dwell_map, int max_iterations, int width_dwell_map, int height_dwell_map)
{
    int img_x = threadIdx.x + blockDim.x * blockIdx.x;
    int img_y = threadIdx.y + blockDim.y * blockIdx.y;

    int dwell;
    float w;

    if (img_x < width_dwell_map && img_y < height_dwell_map)
    {
        dwell = d_dwell_map[img_y * width_dwell_map + img_x];

        if (dwell < max_iterations)
        {
            // temp, just set white tone for now
            d_pixels[img_y * width_dwell_map + img_x] = (float)dwell / (float)max_iterations;;
        }
        else
        {
            d_pixels[img_y * width_dwell_map + img_x] = 0.0f;
        }
    }
}

// -- functions to allow access to CUDA from app --

void setImageSize(int w, int h)
{
    imageParams.width = w;
    imageParams.height = h;
}

void initCUDA()
{
    printf("Initializing CUDA ... ");

    // make sure the hardware is CUDA compatible
    int deviceCount = 0;
    CUDA_CHECK_RETURN(cudaGetDeviceCount(&deviceCount));

    if (deviceCount == 0)
    {
        printf("\nCUDA is not supported on this machine!\n");
        exit(EXIT_FAILURE);
    }

    CUDA_CHECK_RETURN(cudaSetDevice(0)); // use the first device available

    printf("done.\n");
}

void allocCUDA()
{
    printf("Allocating host memory ... ");
    hostBuffers.h_dwell_map = new unsigned int[imageParams.width * imageParams.height];
    hostBuffers.h_image_colours = new float[imageParams.width * imageParams.height]; // black and white, for now
    printf("done.\n");

    printf("Allocating device memory ... ");
    CUDA_CHECK_RETURN(
        cudaMalloc((void**) &deviceBuffers.d_dwell_map, sizeof(unsigned int) * imageParams.width * imageParams.height)
    );
    CUDA_CHECK_RETURN(
        cudaMalloc((void**) &deviceBuffers.d_image_colours, sizeof(unsigned int) * imageParams.width * imageParams.height) // black and white, for now
    );
    printf("done.\n");
}

void runCUDA()
{
    
}

void deallocCUDA()
{
    CUDA_CHECK_RETURN(cudaFree((void*) deviceBuffers.d_dwell_map));
    CUDA_CHECK_RETURN(cudaFree((void*) deviceBuffers.d_image_colours));
    CUDA_CHECK_RETURN(cudaDeviceReset());

    delete[] hostBuffers.h_dwell_map;
    delete[] hostBuffers.h_image_colours;
}
