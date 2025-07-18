#include "CudaMandelbrot.h"
#include "OpenGLMandelbrot.h"

#include <stdio.h>

#define APP_NAME "CudaFractal2D"

void run()
{
	OpenGLMandelbrot openGL(APP_NAME);
	CudaMandelbrot fractalGenerator(INIT_WINDOW_WIDTH, INIT_WINDOW_HEIGHT, 500, openGL.getPBO());

	fractalGenerator.runCUDA();
	// by now there is data in the GPU ready for opengl to display

	openGL.render();
}

int main()
{
	printf("-- %s --\n", APP_NAME);

	run();

	printf("\n-- End of program --\n");
	return 0;
}
