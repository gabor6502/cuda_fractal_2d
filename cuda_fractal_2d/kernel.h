#pragma once

void setImageSize(int, int);
void setIterations(int);

void initCUDA();
void allocCUDA();
void runCUDA();
void deallocCUDA();
