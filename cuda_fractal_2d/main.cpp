#include "CudaMandelbrot.h"
#include "OpenGLMandelbrot.h"

#include <stdio.h>

#define APP_NAME "CudaFractal2D"

int main()
{
	printf("-- %s --\n", APP_NAME);

	CudaMandelbrot fractalGenerator(INIT_WINDOW_WIDTH, INIT_WINDOW_HEIGHT, 500);
	OpenGLMandelbrot openGL(APP_NAME);

	fractalGenerator.runCUDA();
	// by now there is data in the GPU ready for opengl to display
	
	openGL.render();

	printf("\n-- End of program --\n");
	return 0;
}