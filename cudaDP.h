// 
// Joseph Zhong
// josephz@cs.washington.edu
// 19 May 2017
// CSE 599I: Final Project
// Instructor Tanner Schmidt
// Exploring Dynamic Parallism in CUDA C with Mandelbrot Sets
// 
// cudaDP.h
// ---
// 
//  This is the CUDA C implementation with Dynamic Parallelism. 
//

#ifndef CUDA_DP_H
#define CUDA_DP_H

#include <png.h>

#include "cudaCommon.h"

#define BLOCK_SIZE 64
#define DIVIDE_FACTOR 4
#define MAX_DWELL 512
#define MAX_DEPTH 4
#define MIN_SIZE 32

__device__ int pixel_dwell(int w, int h, complexNum cmin, complexNum cmax, int x, int y);

__device__ int same_dwell(int d1, int d2);

__device__ int border_dwell (int w, int h, complexNum cmin, complexNum cmax, int x0, int y0, int d);

__host__ void cudaDPMandelbrotSets(int height, int width, int maxIterations, 
    const float zoom, const float yPos, const float xPos, const float radius,
    const char *filename);

__device__ int commonValue(int v0, int v1, int maxIterations);

__device__ int calculateBorder(int width, int height, int maxIterations,
    complexNum cMin, complexNum cMax, int x0, int y0, int size);

__global__ void pixelKernel(int width, int height, int maxIterations, 
    complexNum cMin, complexNum cMax, int x0, int y0, int size, int *d_output);

__global__ void fillKernel(int width, int x0, int y0, int size, int value, int *d_output); 

__global__ void cudaDPMandelbrotSetsKernel(int height, int width, int maxIterations,
    complexNum cMin, complexNum cMax, int x0, int y0, int size, int depth, const float radius, 
    int *d_output, long long int *d_operations);

#endif // CUDA_DP_H

