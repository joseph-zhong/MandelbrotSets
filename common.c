// 
// Joseph Zhong
// josephz@cs.washington.edu
// 19 May 2017
// CSE 599I: Final Project
// Instructor Tanner Schmidt
// Exploring Dynamic Parallism in CUDA C with Mandelbrot Sets
// 
// common.c 
// ---
//
//  Common code.
//

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <assert.h>

#include "defaults.h"
#include "common.h"

void color(int red, int green, int blue, FILE *fp) {
  unsigned char colors[3];
  colors[0] = (char) red;
  colors[1] = (char) green;
  colors[2] = (char) blue;
  fwrite(colors, 3, 1, fp);
}

void parseArgs(int argc, char *argv[], 
    int *width, int *height, int *maxIterations, 
    char **kernel, char **filename) {
  if (VERBOSE) {
    printf(USAGE, argv[0]);
    printf(EXAMPLE, argv[0]);
  }
  // Set default operation arguments if needed. Parse cmdline arguments
  // otherwise. 
  if (argc != 6) {
    // Set default operational values.
    *width = WIDTH_DEFAULT;
    *height = HEIGHT_DEFAULT;
    *maxIterations = MAX_ITER_DEFAULT;
    *kernel = NAIVE_HOST;
    *filename = NAIVE_OUT_DEFAULT;
  }
  else {
    *width = atoi(argv[1]);
    *height = atoi(argv[2]);
    *maxIterations = atoi(argv[3]);
    *kernel = argv[4];
    *filename = argv[5];
  }
  
  assert(*width > 0 && "Passed width must be positive.\n");
  assert(*height > 0 && "Passed height must be positive.\n");
  assert(*maxIterations > 0 && "Passed maxIterations must be positive.\n");
  assert(strcmp(*kernel, NAIVE_HOST) == 0 || strcmp(*kernel, CUDA_NAIVE) == 0 || strcmp(*kernel, CUDA_DP) == 0);
//  assert(access(*filename, W_OK) == 0 && "Passed output filename could not be opened for writing.\n");
}

