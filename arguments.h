// 
// Joseph Zhong
// josephz@cs.washington.edu
// 19 May 2017
// CSE 599 Final Project
// Instructor Tanner Schmidt
// Exploring Dynamic Parallism in CUDA C with Mandelbrot Sets
// 
// arguments.h
// ---

#ifndef ARGUMENTS_H
#define ARGUMENTS_H

#include "defaults.h"

#include <argp.h>
#include <stdlib.h>

static char doc[] = "Exploring Mandelbrot Sets with CUDA C and Dynamic Parallelism.";
static char args_doc[] = "[WIDTH] [HEIGHT] [MAX_ITER] [KERNEL] [OUTPUT]";
static struct argp_option options[] = { 
    { "width", 'w', "WIDTH", 0, "Width of image to produce."},
    { "height", 'h', "HEIGHT", 0, "Height of image to produce."},
    { "maxIter", 'm', "MAX_ITER", 0, "Maximum iterations to compute before assigning pixel value."},
    { "kernel", 'k', "KERNEL", 0, "Kernel to utilize. Use 'naive', 'cudaNaive', or 'cudaDP'."},
    { "output", 'o', "OUTPUT", 0, "Filename to output image. Will not output if not provided."},
    { "xMin", 'a', "X_MIN", 0, "The minimum X-Value to plot in the image."},
    { "xMax", 'b', "X_MAX", 0, "The maximum X-Value to plot in the image."},
    { "yMin", 'c', "Y_MIN", 0, "The minimum Y-Value to plot in the image."},
    { "yMax", 'd', "Y_MAX", 0, "The maximum Y-Value to plot in the image."},
    { 0 } 
};

struct arguments {
  int width;
  int height;
  int maxIter;
  char *kernel;
  char *output;
  float xMin;
  float xMax;
  float yMin;
  float yMax;
};

void setDefaultArgs(struct arguments *args);

error_t parse_opt(int key, char *arg, struct argp_state *state);

#endif // ARGUMENTS_H

