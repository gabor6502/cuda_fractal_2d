#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include "OpenGLMandelbrot.h"

#include <stdlib.h>
#include <stdio.h>

/*
void framebuffer_size_callback(GLFWwindow* window, int width, int height)
{
	glViewport(0, 0, width, height);
}*/

OpenGLMandelbrot::OpenGLMandelbrot(const char * app_name)
{
	printf("OpenGL init ... ");

	glfwInit();
	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 4);
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

	window = glfwCreateWindow(INIT_WINDOW_WIDTH, INIT_WINDOW_HEIGHT, app_name, NULL, NULL);
	if (window == NULL)
	{
		printf("Couldn't create openGl window.\n");
		glfwTerminate();
		exit(EXIT_FAILURE);
	}
	glfwMakeContextCurrent(window);

	if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress))
	{
		printf("Couldn't initialize GLAD.\n");
		exit(EXIT_FAILURE);
	}

	//glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);
	glViewport(0, 0, INIT_WINDOW_WIDTH, INIT_WINDOW_HEIGHT);

	printf("done.\n");

	// create and bind to a pixel unpack buffer, this is where we'll put CUDA results and have them "unpacked" to the framebuffer
	printf("Generating PBO ... ");
	glGenBuffers(1, &pbo);
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);
	glBufferData(GL_PIXEL_UNPACK_BUFFER, INIT_WINDOW_WIDTH * INIT_WINDOW_HEIGHT, NULL, GL_STREAM_DRAW);
	printf("done.\n");
}

void OpenGLMandelbrot::render()
{
	while (!glfwWindowShouldClose(window))
	{



		glfwSwapBuffers(window);
		glfwPollEvents();
	}
}

OpenGLMandelbrot::~OpenGLMandelbrot()
{
	glDeleteBuffers(1, &pbo);
	glfwTerminate();
}