# MandelbrotSets
Mandelbrot Sets with Dynamic Parallelism in CUDA C

## Build

A device of compute capability of at least 3.5 is required.

### Install 

```
make
```

### Run

```
make run
```

Try some examples

```
./main -w 10000 -h 10000 -m 256 -k cudaDP -o output/cudaDP_10k.png 
```

## Usage 

```
Usage: main [-?V] [-a X_MIN] [-b X_MAX] [-c Y_MIN] [-d Y_MAX] [-h HEIGHT]
            [-k KERNEL] [-m MAX_ITER] [-o OUTPUT] [-w WIDTH] [--xMin=X_MIN]
            [--xMax=X_MAX] [--yMin=Y_MIN] [--yMax=Y_MAX] [--height=HEIGHT]
            [--kernel=KERNEL] [--maxIter=MAX_ITER] [--output=OUTPUT]
            [--width=WIDTH] [--help] [--usage] [--version]

Exploring Mandelbrot Sets with CUDA C and Dynamic Parallelism.

  -a, --xMin=X_MIN           The minimum X-Value to plot in the image.
  -b, --xMax=X_MAX           The maximum X-Value to plot in the image.
  -c, --yMin=Y_MIN           The minimum Y-Value to plot in the image.
  -d, --yMax=Y_MAX           The maximum Y-Value to plot in the image.
  -h, --height=HEIGHT        Height of image to produce.
  -k, --kernel=KERNEL        Kernel to utilize. Use 'naive', 'cudaNaive', or
                             'cudaDP'.
  -m, --maxIter=MAX_ITER     Maximum iterations to compute before assigning
                             pixel value.
  -o, --output=OUTPUT        Filename to output image. Will not output if not
                             provided.
  -w, --width=WIDTH          Width of image to produce.
  -?, --help                 Give this help list
      --usage                Give a short usage message
  -V, --version              Print program version

Mandatory or optional arguments to long options are also mandatory or optional
for any corresponding short options.

Report bugs to josephz@cs.washington.edu.
```
