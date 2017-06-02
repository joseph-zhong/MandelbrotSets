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
//  This is the main program which launches computation kernels.
//

#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <stdbool.h>
#include <time.h>
#include <string.h>

#include "arguments.h"
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
  char *output;
  float xMin;
  float xMax;
  float yMin;
  float yMax;
  float xPos;
  float yPos;

  // Set default arguments.
  struct arguments args;
  setDefaultArgs(&args);

  // Parse custom argument options.
  struct argp argp = { options, parse_opt, args_doc, doc, 0, 0, 0 };
  argp_parse(&argp, argc, argv, 0, 0, &args);

  width = args.width;
  height = args.height;
  maxIterations = args.maxIter;
  kernel = args.kernel;
  output = args.output;
  xMin = args.xMin;
  xMax = args.xMax;
  yMin = args.yMin;
  yMax = args.yMax;
  
  xPos = xMin + (xMax - xMin) / 2;
  yPos = yMin + (yMax - yMin) / 2;

  if (VERBOSE) {
    printf("\n[main] OPERATING PARAMETERS\n");
    printf("-----------------------------\n");
    printf("\t(%0.3f, %0.3f) to (%0.3f, %0.3f)\n", xMin, yMin, xMax, yMax);
    printf("\twidth: '%d'\n\theight: '%d'\n\tmaxIterations: '%d'\n\tkernel: '%s'\n\toutput: '%s'\n\n",
        width, height, maxIterations, kernel, output);
  }
  
  // REVIEW josephz: These could be cmdline arguments but in order to
  // standardize the experiments, we will keep these constant for now.
  const float radius = RADIUS_DEFAULT;
  const complexNum cMin = complexNum(xMin, yMin);
  const complexNum cMax = complexNum(xMax, xMax);

  if (strcmp(kernel, NAIVE_HOST) == 0) {
    if (VERBOSE) {
      printf("[main] Running NAIVE_HOST\n\n");
    }
    naiveMandelbrotSets(height, width, maxIterations, radius, 
        cMin, cMax, output);
  }
  if (strcmp(kernel, CUDA_NAIVE) == 0) {
    if (VERBOSE) {
      printf("[main] Running CUDA_NAIVE\n\n");
    }
    cudaNaiveMandelbrotSets(height, width, maxIterations, radius, cMin, cMax, output);
  }
  if (strcmp(kernel, CUDA_DP) == 0) {
    if (VERBOSE) {
      printf("[main] Running CUDA_DP\n\n");
    }
    cudaDPMandelbrotSets(height, width, maxIterations, radius, cMin, cMax, xPos, yPos, output);
  }

  if (VERBOSE) {
    printf("[main] Time in seconds: ");
    outputTime();
    printf("\n");
  }
  return EXIT_SUCCESS;
}

