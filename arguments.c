// 
// Joseph Zhong
// josephz@cs.washington.edu
// 19 May 2017
// CSE 599 Final Project
// Instructor Tanner Schmidt
// Exploring Dynamic Parallism in CUDA C with Mandelbrot Sets
// 
// arguments.c 
// ---

#include "arguments.h"

const char *argp_program_version = "MandelbrotSets 0.0.0";
const char *argp_program_bug_address = "josephz@cs.washington.edu";

void setDefaultArgs(struct arguments *args) {
  args->width = WIDTH_DEFAULT;
  args->height = HEIGHT_DEFAULT;
  args->maxIter = MAX_ITER_DEFAULT;
  args->kernel = NAIVE_HOST;
  args->output = NULL;
}

error_t parse_opt(int key, char *arg, struct argp_state *state) {
  struct arguments *arguments = (struct arguments*) state->input;
  switch (key) {
  case 'w': 
    arguments->width = atoi(arg);
    break;
  case 'h':
    arguments->height = atoi(arg);
    break;
  case 'm':
    arguments->maxIter = atoi(arg);
    break;
  case 'k':
    arguments->kernel = arg;
    break;
  case 'o':
    arguments->output = arg;
    break;
  case ARGP_KEY_ARG: return 0;
  default: return ARGP_ERR_UNKNOWN;
  }   
  return 0;
}


