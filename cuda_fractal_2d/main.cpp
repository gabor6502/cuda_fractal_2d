#include "kernel.h"
#include <stdio.h>

int main()
{
	printf("Starting CudaFractal2D");

	initCUDA();
	allocCUDA();
	setImageSize(800, 800);
	setIterations(500);
	runCUDA();
	deallocCUDA();

	printf("\nEnd of program\n");
	return 0;
}