###
# Joseph Zhong
# josephz@cs.washington.edu
# 19 May 2017
# CSE 599 Final Project
# Instructor: Tanner Schmidt
# Exploring Dynamic Parallelism in CUDA C with Mandelbrot Sets
# 
# makefile
# ---
#  This handles compilation for the project. 
#
###

FLAGS=-g -Wall -Wpedantic -O4 -lm


all: main 

main: main.o naive.o common.o # cudaNaive.o cudaDP.o
	g++ $(FLAGS) metrics.o common.o naive.o main.o -o main
	./main
	xdg-open images/naive.ppm

main.o: main.c common.o
	g++ $(FLAGS) common.o -c main.c

cudaNaive.o: cudaNaive.cu metrics.o
	nvcc $(FLAGS) metrics.o cudaNaive.cu -o cudaNaive

cudaDP.o: cudaDP.cu metrics.o
	nvcc $(FLAGS) metrics.o cudaDP.cu -o cudaDP

naive.o: naive.c metrics.o common.o
	g++ $(FLAGS) metrics.o common.o -c naive.c

metrics.o: metrics.c
	g++ $(FLAGS) -c metrics.c 

common.o: common.c
	g++ $(FLAGS) -c common.c

clean:
	find \( -name '*.out' -or -name '*.o' -or -name '*~' -or -name '*.ppm' \) -delete
	rm main

