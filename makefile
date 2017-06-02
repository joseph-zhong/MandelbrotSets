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

FLAGS=-g -lm -lpng  
NVCC_FLAGS=-arch=sm_35 -rdc=true -lcudadevrt -Xcompiler -fopenmp 


all: main 

main: arguments.o metrics.o common.o naive.o cudaNaive.o cudaDP.o main.cu 
	mkdir -p output
	nvcc $(FLAGS) $(NVCC_FLAGS) arguments.o metrics.o common.o naive.o cudaNaive.o cudaDP.o main.cu -o main
	
cudaNaive.o: cudaNaive.cu metrics.o
	nvcc $(FLAGS)  $(NVCC_FLAGS) metrics.o -c cudaNaive.cu

cudaDP.o: cudaDP.cu common.o metrics.o
	nvcc $(FLAGS)  $(NVCC_FLAGS) common.o metrics.o -c cudaDP.cu

naive.o: naive.cu metrics.o common.o
	nvcc $(FLAGS) $(NVCC_FLAGS) metrics.o common.o -c naive.cu

metrics.o: metrics.c
	g++ $(FLAGS) -c metrics.c 

common.o: common.cu
	nvcc $(FLAGS) $(NVCC_FLAGS) -c common.cu

arguments.o: arguments.c
	g++ $(FLAGS) -c arguments.c

clean:
	find \( -name '*.out' -or -name '*.o' -or -name '*~' -or -name '*.ppm' \) -delete
	rm main
	rm -r output

run:
	./main 10000 10000 256 cudaDP output/cudaDP.png
	./main 1024 1024 512 cudaNaive output/cudaNaive.png 
	./main 1024 1024 512 naive output/naive.png
	gpicview output/cudaDP.png

