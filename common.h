// 
// Joseph Zhong
// josephz@cs.washington.edu
// 19 May 2017
// CSE 599I: Final Project
// Instructor Tanner Schmidt
// Exploring Dynamic Parallism in CUDA C with Mandelbrot Sets
// 
// common.h 
// ---
//
//  Common code.
//

#ifndef COMMON_H
#define COMMON_H


// Writes RGB to a ppm FILE*. 
void color(int red, int green, int blue, FILE *fp);

// Parses operation parameters.
void parseArgs(int argc, char *argv[], int *width, int *height, int *maxIterations, char **kernel, char **filename);

#endif //COMMON_H

