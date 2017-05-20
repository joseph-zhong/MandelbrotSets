// 
// Joseph Zhong
// josephz@cs.washington.edu
// 19 May 2017
// CSE 599I: Final Project
// Instructor: Tanner Schmidt
// Exploring Dynamic Parallelism in CUDA C with Mandelbrot Sets.
//
// naive.c
// --- 
//  
//  This is a naive C implementation.
//
//  Each iteration the newZ is calculated from the square of the prevZ summed
//  with the current pixel value. We begin at the origin.
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
    printf("\twidth: '%d'\n\theight: '%d'\n\tmaxIterations: '%d'\n\tfilename: '%s'\n\n",
        width, height, maxIterations, filename);
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
    naiveMandelbrotSets(y, x, height, width, maxIterations, zoom, yPos, xPos, radius, fp);
  }
  
  reportClock();
  reportOperations();
  reportFlops();
  return EXIT_SUCCESS;
}

