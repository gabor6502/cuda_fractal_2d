#include "kernel.h"
#include <stdio.h>

int main()
{
	printf("-- CudaFractal2D --\n");

	initCUDA();
	allocCUDA();
	setImageSize(800, 800);
	setIterations(500);
	runCUDA();
	deallocCUDA();

	printf("\n-- End of program --\n");
	return 0;
}