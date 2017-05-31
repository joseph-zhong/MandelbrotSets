// 
// Joseph Zhong
// josephz@cs.washington.edu
// 19 May 2017
// CSE 599I: Final Project
// Instructor: Tanner Schmidt
// Exploring Dynamic Parallelism in CUDA C with Mandelbrot Sets.
//
// metrics.c
// ---
//  
//  This class will handle metrics measuring of the various experiments.
//

#include <stdio.h>
#include <stdlib.h>

#include "metrics.h"
#include "defaults.h"

// Define extern variables.
double g_time_spent = 0.0;
long long int g_operations = 0;

void endClock(clock_t start) {
  g_time_spent += (double) (clock() - start) / CLOCKS_PER_SEC;
}

void reportClock() {
  printf("Clock Report\n");
  printf("------------\n");
  printf("\tTotal Time: %0.3f seconds\n\n", g_time_spent);
}

void reportOperations() {
  printf("Operations Report\n");
  printf("-----------------\n");
  printf("\tTotal Operations: %lld\n\n", g_operations);
}

void reportFlops() {
  printf("Flops Report\n");
  printf("------------\n");
  printf("\tFlops: %0.3f\n\n", ((double) g_operations / max(0.000001, g_time_spent)));
}

void reportConcise(const int width, const int height, const int maxIterations, 
    const char *kernel, const char *filename) {
  printf("[Res=%d] [Iter=%d] [Kernel=%s] [Seconds=%0.3e] [MP/s=%0.3e]\n", 
      width, maxIterations, kernel, g_time_spent, (float) width * height / g_time_spent);
}

void outputTime() {
  printf("%0.3e", g_time_spent);
}

