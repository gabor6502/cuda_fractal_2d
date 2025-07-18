#include "kernel.h"
#include "graphics.h"

#include <stdio.h>

#define APP_NAME "CudaFractal2D"

int main()
{
	printf("-- %s --\n", APP_NAME);

	GLFWwindow* window = initOpenGL(APP_NAME);
	
	// render

	while (!glfwWindowShouldClose(window))
	{


		glfwSwapBuffers(window);
		glfwPollEvents();
	}

	/*
	initCUDA();
	allocCUDA();
	setImageSize(800, 800);
	setIterations(500);
	runCUDA();
	deallocCUDA();
	*/

	glfwTerminate();

	printf("\n-- End of program --\n");
	return 0;
}