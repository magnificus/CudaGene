#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size);



__global__ void addKernel(int *c, const int *a, const int *b)
{
    int i = threadIdx.x;
    c[i] = a[i] - b[i];
}


extern "C" void
launch_addkernel(int size, int *c, const int *a, const int *b) {
    addKernel << <1, size >> > (c, a, b);
}

