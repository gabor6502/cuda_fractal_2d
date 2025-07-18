#include "kernel.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

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


void runCUDA()
{
    
}

void deallocCUDA()
{
    CUDA_CHECK_RETURN(cudaDeviceReset());
}
