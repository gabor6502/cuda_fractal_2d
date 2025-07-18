#include "kernel.h"

int main()
{
	initCUDA();
	
	runCUDA();

	deallocCUDA();

	return 0;
}