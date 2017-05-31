// 
// Joseph Zhong
// josephz@cs.washington.edu
// 19 May 2017
// CSE 599I: Final Project
// Instructor: Tanner Schmidt
// Exploring Dynamic Parallelism in CUDA C with Mandelbrot Sets.
//
// metrics.h
// ---
//  
//  This class will handle metrics measuring of the various experiments.
//

#ifndef METRICS_H
#define METRICS_H

#include <time.h>

#define max(a,b)						 \
({ 													 \
	__typeof__ (a) _a = (a);	 \
	__typeof__ (b) _b = (b);	 \
	_a > _b ? _a : _b; 				 \
})													


// Global time spent.
extern double g_time_spent;

// Global operations computed.
extern long long int g_operations;

// This takes a start time and adds to the total time spent. 
void endClock(clock_t start);

// This reports the total time spent.
void reportClock();

// This reports the total operations computed.
void reportOperations();

// This reports Flops.
void reportFlops();

void reportConcise(const int width, const int height, const int maxIterations, 
        const char *kernel, const char *filename);

void outputTime();

#endif // METRICS_H

