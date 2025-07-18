#pragma once

/*
 * All CUDA calls and fractal generation handled here
 */

// maximum magnitude of an element of the mandelbrot set, squared
#define MAX_MAG_SQ 4

// use first device found
#define DEVICE 0

#define INIT_ITERATIONS 500

void setImageSize(int width, int height);
void setIterations(int iterations);

void initCUDA();
void allocCUDA();
void runCUDA();
void deallocCUDA();

//void initCudaOpenGLInterop(unsigned int pbo);

