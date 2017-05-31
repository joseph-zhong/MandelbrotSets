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
  int maxIterations;
  char *kernel;
  char *filename;

  parseArgs(argc, argv, &width, &height, &maxIterations, &kernel, &filename);

  if (VERBOSE) {
    printf("\n[main] OPERATING PARAMETERS\n");
    printf("-----------------------------\n");
    printf("\twidth: '%d'\n\theight: '%d'\n\tmaxIterations: '%d'\n\tkernel: '%s'\n\tfilename: '%s'\n\n",
        width, height, maxIterations, kernel, filename);
  }

  // REVIEW josephz: These could be cmdline arguments but in order to
  // standardize the experiments, we will keep these constant for now.
  const float radius = RADIUS_DEFAULT;
  const complexNum cMin = complexNum(X_MIN_DEFAULT, Y_MIN_DEFAULT);
  const complexNum cMax = complexNum(X_MAX_DEFAULT, Y_MAX_DEFAULT);

  if (strcmp(kernel, NAIVE_HOST) == 0) {
    if (VERBOSE) {
      printf("[main] Running NAIVE_HOST\n\n");
    }
    naiveMandelbrotSets(height, width, maxIterations, radius, 
        cMin, cMax, filename);
  }
  if (strcmp(kernel, CUDA_NAIVE) == 0) {
   cudaNaiveMandelbrotSets(height, width, maxIterations, radius, cMin, cMax, filename);
  }
  if (strcmp(kernel, CUDA_DP) == 0) {
    cudaDPMandelbrotSets(height, width, maxIterations, radius, cMin, cMax, filename);
  }

  outputTime();
  return EXIT_SUCCESS;
}

