// 
// Joseph Zhong
// josephz@cs.washington.edu
// 19 May 2017
// CSE 599I: Final Project
// Instructor: Tanner Schmidt
// Exploring Dynamic Parallelism in CUDA C with Mandelbrot Sets.
//
// main.cu
// ---
//  This is the main program which launches computer kernels.
//

#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <stdbool.h>
#include <time.h>
#include <string.h>

#include "cudaNaive.h"
#include "common.h"
#include "defaults.h"
#include "metrics.h"
#include "naive.h"
#include "cudaNaive.h"
#include "cudaDP.h"


int main(int argc, char *argv[]) {
  // Default operational values.
  int width;
  int height;
  int x;
  int y;
  int maxIterations;
  char *kernel;
  char *filename;
  FILE *fp;

  parseArgs(argc, argv, &width, &height, &maxIterations, &kernel, &filename);

  if (VERBOSE) {
    printf("\n[main] OPERATING PARAMETERS\n");
    printf("-----------------------------\n");
    printf("\twidth: '%d'\n\theight: '%d'\n\tmaxIterations: '%d'\n\tkernel: '%s'\n\tfilename: '%s'\n\n",
        width, height, maxIterations, kernel, filename);
  }

  // Set filename for output image.
  fp = fopen(filename, "wb");

  // REVIEW josephz: These could be cmdline arguments but in order to
  // standardize the experiments, we will keep these constant for now.
  const float zoom = ZOOM_DEFAULT;
  const float xPos = X_POS_DEFAULT;
  const float yPos = Y_POS_DEFAULT;
  const float radius = RADIUS_DEFAULT;

  double pr, pi;                       //real and imaginary part of the pixel p
  double newRe, newIm, oldRe, oldIm;   //real and imaginary parts of new and old z

  // Write header to ppm file.
  fprintf(fp, "P6\n# Mandelbrot Set. \n%d %d\n255\n", width, height);

  if (strcmp(kernel, NAIVE_HOST) == 0) {
    if (VERBOSE) {
      printf("[main] Running NAIVE_HOST\n\n");
    }
    naiveMandelbrotSets(y, x, height, width, maxIterations, zoom, yPos, xPos, radius, fp);
  }
  if (strcmp(kernel, CUDA_NAIVE) == 0) {
    // Host input setup: image and operations count.
    const int OUTPUT_SIZE = sizeof(char) * height * width * 3;
    char *h_output = (char*) malloc(OUTPUT_SIZE);
    long long int *h_operations = (long long int*) calloc(1, sizeof(long long int));

    // Device output setup: image and operations.
    char *d_output;
    long long int *d_operations;
    cudaCheck(cudaMalloc(&d_operations, sizeof(long long int)));
    cudaCheck(cudaMalloc(&d_output, OUTPUT_SIZE));
    // cudaCheck(cudaMemset(d_operations, 0, OUTPUT_SIZE));

    // Set operations to 0.
    cudaCheck(cudaMemcpy(d_operations, h_operations, sizeof(long long int), cudaMemcpyHostToDevice));
    
    // Kernel Size.
    dim3 gridSize(ceil(width / TILE_WIDTH), ceil(height / TILE_WIDTH), 1);
    dim3 blockSize(TILE_WIDTH, TILE_WIDTH, 1);

    // Begin timer.
    //struct timespec tstart={0,0};
    clock_t start = clock();

    // Launch Kernel.
    naiveMandelbrotSetsKernel<<<gridSize, blockSize>>>(
        height, width, maxIterations, zoom, yPos, xPos, radius, d_output, d_operations); 
    cudaDeviceSynchronize();

    // Stop timer.
    endClock(start);

    // Copy output and operations.
    cudaCheck(cudaMemcpy(h_output, d_output, OUTPUT_SIZE, cudaMemcpyDeviceToHost));    
    cudaCheck(cudaMemcpy(h_operations, d_operations, sizeof(long long int), cudaMemcpyDeviceToHost));

    // Free output and operations.
    cudaFree(d_output);
    cudaFree(d_operations);

    fwrite(h_output, OUTPUT_SIZE, 1, fp);
    g_operations = *h_operations;

    free(h_output);
    free(h_operations);
  }
  
  reportClock();
  reportOperations();
  reportFlops();
  return EXIT_SUCCESS;
}

