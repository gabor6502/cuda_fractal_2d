/*
 * All exclusively OpenGL related procedures are managed here 
 */
#pragma once

#define INIT_WINDOW_WIDTH 800
#define INIT_WINDOW_HEIGHT 800

/*
 * Handles the OpenGL side of things, like window management, rendering results, etc.
 */
class GLFWwindow;
class OpenGLMandelbrot
{
private:
	GLFWwindow* window;
	unsigned int pbo;

public:
	OpenGLMandelbrot(const char * app_name);
	~OpenGLMandelbrot();

	void render();

	inline
	unsigned int getPBO() { return pbo; }

	inline
	GLFWwindow * getWindow() { return window; }
};
