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
#include "cudaCommon.h"
#include "defaults.h"
#include "metrics.h"

__host__ void cudaNaiveMandelbrotSets(int height, int width, int maxIterations, const float radius, 
    const char *filename) {
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
	// cudaNaiveMandelbrotSetsKernel<<<gridSize, blockSize>>>(
	// 	 height, width, maxIterations, zoom, yPos, xPos, radius, d_output, d_operations); 
  cudaNaiveMandelbrotSetsKernel<<<gridSize, blockSize>>>(d_output, width, height, maxIterations, radius, complexNum(-1.5, -1), complexNum(0.5, 1));

	// Stop timer.
	endClock(start);

	// Copy output and operations.
	cudaCheck(cudaMemcpy(h_output, d_output, OUTPUT_SIZE, cudaMemcpyDeviceToHost));        

	// Free output and operations.
	cudaFree(d_output);

  // Write to output.
	//fwrite(h_output, OUTPUT_SIZE, 1, fp);
  save_image(filename, h_output, width, height, maxIterations);

	free(h_output);
}

// __global__ void cudaNaiveMandelbrotSetsKernel(int height, int width, int maxIterations, 
//      const float zoom, const float yPos, const float xPos, const float radius, 
//      int *d_output, long long int *d_operations) {
//   double newRe, newIm, oldRe, oldIm, pr, pi; 
  
//   // Naively iterate through each pixel.
//   int y = threadIdx.y + blockDim.y * blockIdx.y;
//   int x = threadIdx.x + blockDim.x * blockIdx.x;
  
//   // BUGBUG josephz: A column of pixels on the right hand side seems to have
//   // been lost, either as a black bar, or distorted white and black noise.
//   if (x >= width || y >= height) return;
   
//   // Calculate Z from the pixel location, zoom, and position values.
//   pr = 1.5 * (x - width / 2) / (0.5 * zoom * width) + xPos; // 8 Ops.
//   pi = (y - height / 2) / (0.5 * zoom * height) + yPos;     // 7 Ops.
//   newRe = newIm = oldRe = oldIm = 0; // 4 Ops. 
//   int i;
//   for(i = 0; i < maxIterations; i++) {  // 3 Ops.
//     oldRe = newRe;
//     oldIm = newIm;
//     newRe = oldRe * oldRe - oldIm * oldIm + pr; 
//     newIm = 2 * oldRe * oldIm + pi; // 11 Ops.
//     // Stop once our point exceeds the target radius.
//     if((newRe * newRe + newIm * newIm) > radius) break; // 4 Ops.
//   }   

//   // 43 Ops.  

//   // If iteration limit is reached, fill black. Colored otherwise.
//   int output_index = width * y + x;
//   if(i == maxIterations) {
//     d_output[output_index] = 0;
//   }   
//   else {
//     double z = sqrt(newRe * newRe + newIm * newIm);
//     int brightness = 256.0 * log2(1.75 + i - log2(log2(z))) / log2((double)maxIterations);
//     d_output[output_index] = brightness;
//   }
//   *d_operations += 56;
// }


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
