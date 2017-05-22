// 
// Joseph Zhong
// josephz@cs.washington.edu
// 19 May 2017
// CSE 599I: Final Project
// Instructor Tanner Schmidt
// Exploring Dynamic Parallism in CUDA C with Mandelbrot Sets
// 
// cudaDP.cu
// ---
// 
//  This is the CUDA C implementation with Dynamic Parallelism. 
//

#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <stdbool.h>
#include <time.h>
#include <png.h>

#include "cudaDP.h"
#include "common.h"
#include "cudaCommon.h"
#include "defaults.h"
#include "metrics.h"

#define INITSPLIT 32
#define MAX_DEPTH 4
#define MIN_SIZE 32

__host__ void cudaDPMandelbrotSets(int height, int width, int maxIterations, 
    const float zoom, const float yPos, const float xPos, const float radius, 
    FILE *fp) {
  const int OUTPUT_SIZE = height * width * sizeof(int);
  int *h_output = (int*) malloc(OUTPUT_SIZE);
  long long int *h_operations = (long long int*) calloc(1, sizeof(long long int));

  // int *d_output = NULL; 
  int *d_output = NULL; 
  long long int *d_operations = NULL;
  cudaCheck(cudaMalloc((void **) &d_output, OUTPUT_SIZE));
  cudaCheck(cudaMalloc((void **) &h_output, sizeof(long long int)));

  cudaCheck(cudaMemcpy(d_operations, h_operations, sizeof(long long int), cudaMemcpyHostToDevice));

  dim3 blockSize(64, 4);
  dim3 gridSize(32, 32);

  clock_t start = clock();

  cudaDPMandelbrotSetsKernel<<<gridSize, blockSize>>>(height, width, maxIterations,
      complexNum(-1.5, -1), complexNum(0.5, 1), 0, 0, width / 32, 1,
      d_output, d_operations);
  cudaCheck(cudaThreadSynchronize());
  
  endClock(start);
  
  cudaCheck(cudaMemcpy(h_output, d_output, OUTPUT_SIZE, cudaMemcpyDeviceToHost));
  cudaCheck(cudaMemcpy(h_operations, d_operations, sizeof(long long int), cudaMemcpyDeviceToHost));

 	// Free device output and operations.
  cudaFree(d_output);
  cudaFree(d_operations);
 
	// Write output and operations.
  // fwrite(h_output, OUTPUT_SIZE, 1, fp);
  // save_image("images/cudaDP.png", h_output, width, height);
  g_operations = *h_operations;
 

  free(h_output);
  free(h_operations);
}

__device__ int commonValue(int v0, int v1, int maxIterations) {
  if (v0 == v1) {
    return v0;
  }
  if (v0 == maxIterations + 1 || v1 == maxIterations + 1) {
    return min(v0, v1);
  }
  return -1;
}

__device__ int calculateBorder(int width, int height, int maxIterations,
    complexNum cMin, complexNum cMax, int x0, int y0, int size, const float radius) {
  int tIdx = threadIdx.y * blockDim.x + threadIdx.x;
  int blockSize = blockDim.x * blockDim.y;
  int value = maxIterations + 1;
  for (int pixel = tIdx; pixel < size; pixel += blockSize) {
    for (int boundary = 0; boundary < 4; ++boundary) {
      int x = boundary % 2 != 0 ? x0 + pixel : (boundary == 0 ? x0 + size - 1 : x0); 
      int y = boundary % 2 == 0 ? y0 + pixel : (boundary == 1 ? y0 + size - 1 : y0);
      value = calculatePixelValue(width, height, maxIterations, cMin, cMax, x, y, radius); 
    }
  }

  __shared__ int s_output[64 * 4];
  int numThreads = min(size, 64 * 4);
  if (tIdx < numThreads) {
    s_output[tIdx] = value;
  }
  __syncthreads();

  while (numThreads > 1) {
    if (tIdx < numThreads / 2) {
      s_output[tIdx] = commonValue(s_output[tIdx], s_output[tIdx + numThreads / 2], maxIterations);
    }
    __syncthreads();
    numThreads /= 2;
  }
  return s_output[0];
}

__global__ void pixelKernel(int width, int height, int maxIterations, 
    complexNum cMin, complexNum cMax, int x0, int y0, int size, const float radius, int *d_output) {
  int x = threadIdx.x + blockDim.x * blockIdx.x;
  int y = threadIdx.y + blockDim.y * blockIdx.y;

  if (x < size && y < size) {
    x += x0;
    y += y0;
    d_output[y * width + x] = calculatePixelValue(width, height, maxIterations, cMin, cMax, x, y, radius);
    // int outputIndex = CHANNELS * width * y + x * CHANNELS;
    // int pixelValue = calculatePixelValue(width, height, maxIterations, cMin, cMax, x, y, radius);
    // d_output[outputIndex] = (char) pixelValue;
    // d_output[outputIndex + 1] = (char) pixelValue;
    // d_output[outputIndex + 2] = (char) 255;
  }
}

__global__ void fillKernel(int width, int x0, int y0, int size, int value, int *d_output) {
  int x = threadIdx.x + blockDim.x * blockDim.y;
  int y = threadIdx.y + blockDim.y * blockDim.y;

  if (x < size && y < size) {
    x += x0;
    y += y0;
    d_output[y * width + x] = value;
    // int outputIndex = CHANNELS * width * y + x * CHANNELS;
    // d_output[outputIndex] = (char) value; 
    // d_output[outputIndex + 1] = (char) value; 
    // d_output[outputIndex + 2] = (char) 255; 
  }
}

__global__ void cudaDPMandelbrotSetsKernel(int height, int width, int maxIterations,
    complexNum cMin, complexNum cMax, int x0, int y0, int size, int depth, const float radius,
    int *d_output, long long int *d_operations) {
  x0 += size * blockIdx.x;
  y0 += size * blockIdx.y;

  // Compute border values. 
  // By the Mariani-Silver Algorithm, we know that we can fill the border's
  // box with the border values if they all match. Otherwise, we can split the
  // box into smaller rectangles and recurse. 
  int borderVal = calculateBorder(width, height, maxIterations, cMin, cMax, x0, y0, size, radius); 

  if (threadIdx.x == 0 && threadIdx.y == 0) {
    if (borderVal != -1) {
      dim3 fillBlockSize(64, 4);
      dim3 fillGridSize(divup(size, 64), divup(size, 4));
      fillKernel<<<fillGridSize, fillBlockSize>>>(width, height, x0, y0, size, d_output);
    }
    else if (depth + 1 < MAX_DEPTH && size / 4 > MIN_SIZE) {
      dim3 recurseGridSize(4, 4);
      dim3 recurseBlockSize(blockDim.x, blockDim.y);
      cudaDPMandelbrotSetsKernel<<<recurseGridSize, recurseBlockSize>>>(height, width, maxIterations, 
          cMin, cMax, x0, y0, size / 4, depth + 1, radius, d_output, d_operations); 
    }
    else {
      dim3 pixelGridSize(divup(size, 64), divup(size, 4));
      dim3 pixelBlockSize(64, 4);
      pixelKernel<<<pixelGridSize, pixelBlockSize>>>(width, height, maxIterations,
          cMin, cMax, x0, y0, size, radius, d_output);
    }
    // cudaCheck(cudaGetLastError());
  }
}


