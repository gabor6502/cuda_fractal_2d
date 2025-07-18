#pragma once

#define INIT_ITERATIONS 500

void setImageSize(int, int);
void setIterations(int);

void initCUDA();
void allocCUDA();
void runCUDA();
void deallocCUDA();
