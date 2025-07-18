#include "kernel.h"
#include "graphics.h"
#include "interopCuGl.h"

#include <stdio.h>

#define APP_NAME "CudaFractal2D"

int main()
{
	printf("-- %s --\n", APP_NAME);

	GLFWwindow* window = initOpenGL(APP_NAME);
	
	// cuda calls in positionn to just run once, testing out to see if everything worked

	initCUDA();
	allocCUDA();
	setImageSize(INIT_WINDOW_WIDTH, INIT_WINDOW_HEIGHT);
	setIterations(INIT_ITERATIONS);

	runCUDA();
	// by now there is data in the GPU ready for opengl to display

	// create and bind to a pixel unpack buffer, this is where we'll put CUDA results and have them "unpacked" to the framebuffer
	GLuint pbo;
	glGenBuffers(1, &pbo);
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);
	//glBufferData(GL_PIXEL_UNPACK_BUFFER, 0, NULL, GL_STREAM_DRAW); // init with anythign

	// - render -

	while (!glfwWindowShouldClose(window))
	{


		glfwSwapBuffers(window);
		glfwPollEvents();
	}


	deallocCUDA();


	glfwTerminate();

	printf("\n-- End of program --\n");
	return 0;
}