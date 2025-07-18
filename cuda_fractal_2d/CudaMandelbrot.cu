#include "cuda_runtime.h"
#include "device_launch_parameters.h"
//#include "cuda_gl_interop.h"
#include "CudaMandelbrot.h"

// warnings for floats being used in cuda/std/complex, just redefine infinity
#define INFINITY std::numeric_limits<double>::infinity()
#include <cuda/std/complex>

#include <stdlib.h>
#include <stdio.h>

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

typedef cuda::std::complex<double> complex;

// starting off with basic alg, no DP/mariani-silver yet

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
    while (dwell++ < max_iterations && z.real() * z.real() + z.imag() * z.imag() < MAX_MAG_SQ)
    {
        z = z * z + c;
    }

    return dwell;
}

__global__ void k_mandelbrot(unsigned int * d_dwell_map, int max_iterations,
                                   int width_dwell_map, int height_dwell_map, 
                                   complex bottom_left, complex top_right)
{
    int img_x = threadIdx.x + blockDim.x * blockIdx.x;
    int img_y = threadIdx.y + blockDim.y * blockIdx.y;

    if (img_x < width_dwell_map && img_y < height_dwell_map)
    {
        d_dwell_map[img_y * height_dwell_map + img_x] 
            = mandelbrot_pixel(max_iterations, width_dwell_map, height_dwell_map, 
                               bottom_left, top_right, 
                                img_x, img_y);
    }

}

// assign colours of the screen from dwell-map

__global__ void k_colour_dwell_map(float * d_pixels, unsigned int * d_dwell_map, int max_iterations, int width_dwell_map, int height_dwell_map)
{
    int img_x = threadIdx.x + blockDim.x * blockIdx.x;
    int img_y = threadIdx.y + blockDim.y * blockIdx.y;

    int dwell;

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

/*
void initCudaOpenGLInterop(unsigned int pbo)
{
    size_t device_buffer_size = sizeof(unsigned int) * imageParams.width * imageParams.height;

    deviceBuffers.pbo = pbo;

    CUDA_CHECK_RETURN(cudaGLSetGLDevice(DEVICE));

    // register the pbo under the resource given (r/w flag assumed)
    CUDA_CHECK_RETURN(
        cudaGraphicsGLRegisterBuffer(deviceBuffers.image_resource, pbo, cudaGraphicsRegisterFlagsNone));

    // map the pbo resource pointer to the device buffer supplied
    CUDA_CHECK_RETURN(
        cudaGraphicsResourceGetMappedPointer((void **)deviceBuffers.d_image_colours, 
                                              &device_buffer_size,
                                              *deviceBuffers.image_resource));
}*/


CudaMandelbrot::CudaMandelbrot(int width, int height, int iter)
    : image_width(width), image_height(height), iterations(iter)
{
    int deviceCount = 0;
    cudaDeviceProp deviceProp;

    printf("Initializing CUDA ... ");

    // make sure the hardware is CUDA compatible
    CUDA_CHECK_RETURN(cudaGetDeviceCount(&deviceCount));

    if (deviceCount == 0)
    {
        printf("\nCUDA is not supported on this machine!\n");
        exit(EXIT_FAILURE);
    }

    CUDA_CHECK_RETURN(cudaSetDevice(DEVICE));

    CUDA_CHECK_RETURN(cudaGetDeviceProperties(&deviceProp, DEVICE));
    max_block_size_x = deviceProp.maxThreadsDim[0];
    max_block_size_y = deviceProp.maxThreadsDim[1];
    max_grid_size_x = deviceProp.maxGridSize[0];
    max_grid_size_y = deviceProp.maxGridSize[1];

    printf("done.\n");

    printf("Allocating host memory ... ");
    h_dwell_map = new unsigned int[image_width * image_height];
    h_image_colours = new float[image_width * image_height]; // black and white, for now
    printf("done.\n");

    printf("Allocating device memory ... ");
    CUDA_CHECK_RETURN(
        cudaMalloc((void**) &d_dwell_map, sizeof(unsigned int)* image_width* image_height)
    );
    CUDA_CHECK_RETURN(
        cudaMalloc((void**) &d_image_colours, sizeof(unsigned int)* image_width* image_height) // black and white, for now
    );
    printf("done.\n");
}

void CudaMandelbrot::runCUDA()
{
    const unsigned int nThrdX = ceil((float)image_width / (float)max_block_size_x);
    const unsigned int nThrdY = ceil((float)image_height / (float)max_block_size_y);
    dim3 blockDimensions = dim3(nThrdX, nThrdY);

    const unsigned int nBlkX = ceil((float)nThrdX / (float)max_grid_size_x);
    const unsigned int nBlkY = ceil((float)nThrdY / (float)max_grid_size_y);
    dim3 gridDimensions = dim3(nBlkX, nBlkY);

    complex bottom_left(-1.5, 1.0);
    complex top_right(0.5, 1.0);

    printf("Executing kernels ... ");

    k_mandelbrot <<<blockDimensions, gridDimensions>>>(d_dwell_map, iterations, image_width, image_height, bottom_left, top_right);

    k_colour_dwell_map <<<blockDimensions, gridDimensions>>>(d_image_colours, d_dwell_map, iterations, image_width, image_height);

    printf("done!\n");
}

CudaMandelbrot::~CudaMandelbrot()
{
    printf("Deallocating device memory ... ");
    CUDA_CHECK_RETURN(cudaFree((void*)d_dwell_map));
    CUDA_CHECK_RETURN(cudaFree((void*)d_image_colours));
    CUDA_CHECK_RETURN(cudaDeviceReset());
    printf("done.\n");

    printf("Deallocating host memory ... ");
    delete[] h_dwell_map;
    delete[] h_image_colours;
    printf("done.\n");
}