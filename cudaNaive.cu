// 
// Joseph Zhong
// josephz@cs.washington.edu
// 19 May 2017
// CSE 599I: Final Project
// Instructor Tanner Schmidt
// Exploring Dynamic Parallism in CUDA C with Mandelbrot Sets
// 
// cudaNaive.cu
// ---
// 
//  This is the naive CUDA C implementation without usage of Dynamic
//  Parallelism. 
//

#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <stdbool.h>
#include <time.h>

#include "cudaNaive.h"
#include "common.h"
#include "defaults.h"
#include "metrics.h"

__host__ void cudaNaiveMandelbrotSets(int height, int width, int maxIterations, const float radius, 
    const complexNum cMin, const complexNum cMax, const char *filename) {
	// Host input setup: image.
	const int OUTPUT_SIZE = sizeof(int) * height * width;
	int *h_output = (int*) malloc(OUTPUT_SIZE);

	// Device output setup: image.
	int *d_output;
	cudaCheck(cudaMalloc(&d_output, OUTPUT_SIZE));

	// Kernel Size.
	dim3 gridSize(ceil(width / TILE_WIDTH), ceil(height / TILE_WIDTH), 1); 
	dim3 blockSize(TILE_WIDTH, TILE_WIDTH, 1); 

	// Begin timer.
	clock_t start = clock();

	// Launch Kernel.
  cudaNaiveMandelbrotSetsKernel<<<gridSize, blockSize>>>(d_output, width, height, maxIterations, radius, 
      cMin, cMax);

	// Stop timer.
	endClock(start);

	// Copy output and operations.
	cudaCheck(cudaMemcpy(h_output, d_output, OUTPUT_SIZE, cudaMemcpyDeviceToHost));        

	// Free output and operations.
	cudaFree(d_output);

  // Write to output.
  save_image(filename, h_output, width, height, maxIterations);

	free(h_output);
}

__global__ void cudaNaiveMandelbrotSetsKernel(int *d_output, 
    int width, int height, int maxIterations, const float radius, 
    complexNum cMin, complexNum cMax) {

  int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
  if (x >= width || y >= height) return;

	int value = calculatePixelValue(width, height, maxIterations,
    cMin, cMax, x, y, radius);
	d_output[y * width + x] = value;
}

