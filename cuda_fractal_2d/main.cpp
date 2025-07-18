#include "kernel.h"
#include "graphics.h"

#include <stdio.h>

#define APP_NAME "CudaFractal2D"

int main()
{
	printf("-- %s --\n", APP_NAME);


	printf("OpenGL init ... ");
	GLFWwindow* window = initOpenGL(APP_NAME);
	printf("done.\n");

	initCUDA();
	//initCudaOpenGLInterop(pbo);
	allocCUDA();
	setImageSize(INIT_WINDOW_WIDTH, INIT_WINDOW_HEIGHT);
	setIterations(INIT_ITERATIONS);

	//GLuint pbo = createAndBindPBO();

	// cuda calls in position to just run once, testing out to see if everything worked




	runCUDA();
	// by now there is data in the GPU ready for opengl to display



	// - render -
	
	while (!glfwWindowShouldClose(window))
	{
		//glUnmapBuffer(GL_PIXEL_UNPACK_BUFFER);

		glfwSwapBuffers(window);
		glfwPollEvents();
	}

	deallocCUDA();
	
	glfwTerminate();

	

	printf("\n-- End of program --\n");
	return 0;
}