# Bilateral Grid Filtering (NPP + CUDA)

## Overview

This repository implements a bilateral filter for single channel grayscale images in two ways:
- Using NVIDIA Performance Primitives (NPP) (nppiFilterBilateralGaussBorder)
- Using a custom GPU bilateral grid [1] implementation (bilateral grid splat/slice kernels)

It is meant as an example of using NPP together with a CUDA custom kernel to compare approaches.

## Repository layout
```
- bin/                - build output (executable: bin/bilateralGrid)
- data/               - input/output images and image list (see below)
  - input/            - input images referenced by the list file
  - output/           - saved results (NPP and grid filtered images)
  - img_list_attribution.txt - newline list of input filenames (one per line)
- src/                - source code (main.cpp, bilateralGrid.cu/h)
- Makefile            - build rules (uses nvcc)
- run.sh              - simple wrapper to run the binary with a default flag
```

## Dependencies
- CUDA Toolkit (nvcc, runtime) with NVIDIA Performance Primitives (NPP) development libraries. Tested with CUDA toolkit 12.8.
- FreeImage (used by helper image I/O in the sample) and image libraries:
    ```bash
    sudo apt-get install libfreeimage3 libfreeimage-dev

    sudo apt-get install libpng-dev libjpeg-dev libtiff-dev
    ```
- CUDA Samples (common utilities) - required for helper_cuda.h and helper_string.h used by the sample. Clone the [CUDA Samples](https://github.com/NVIDIA/cuda-samples) git repository in the same root directory with this project or adjust the paths in Makefile.

## Build
The Makefile compiles the project with nvcc. Edit INCLUDES/LDFLAGS in the Makefile if your CUDA or library locations differ.

To build:
```bash
make
```

To build with debug symbols:
```bash
make dbg=1
```

## Run

A small wrapper script is provided to run the sample with sensible defaults:

- run.sh invokes the binary as:
  ./bin/bilateralGrid -mask_size=32 -filelist="data/img_list_attribution.txt" -input_dir="data/input/" -output_dir="data/output/"

The program can also be run directly. The executable accepts the following command-line options (defaults are shown where applicable):

- -filter_radius=<int>    NPP bilateral filter radius (default: 5; note: run.sh sets this to 32)
- -sigma_v=<float>        NPP bilateral filter value sigma (default: 50.0)
- -sigma_p=<float>        NPP bilateral filter position sigma (default: 50.0)
- -grid_sigma_s=<float>   Bilateral grid spatial sigma (default: 16.0)
- -grid_sigma_r=<float>   Bilateral grid range sigma (default: 32.0)
- -filelist=<path>        Path to the image list file (default: data/img_list_attribution.txt)
- -input_dir=<dir>        Directory containing input images (default: data/input/)
- -output_dir=<dir>       Directory to write output images (default: data/output/)
- -help                   Print usage

Examples:
```bash
# Use the wrapper script (uses mask_size=32 by default)
./run.sh

# Run directly with custom parameters
./bin/bilateralGrid -mask_size=5 -filelist="data/img_list_attribution.txt" -input_dir="data/input/" -output_dir="data/output/"
```

Input list format
Each non-empty line in the file list (default: data/img_list_attribution.txt) should contain an input filename relative to the input directory. For example:
```
lena.png
myphoto.pgm
```

Output files
For each input image with a filename {name} the program writes:
- data/output/{name}_npp_filtered.pgm
- data/output/{name}_grid_filtered.pgm

## Notes
- Make sure NPP and FreeImage libs are present on your linker path. The Makefile links a number of npp* libraries and -lfreeimage; update LDFLAGS if your distro places them elsewhere.
- The example main program creates a CUDA stream and times both the NPP filter and the custom bilateral grid filter for each image in the list.
- The project uses npp::Image helpers for loading/saving images; supported input formats depend on FreeImage.

## Cleaning
To remove built binaries:
```bash
make clean
```

## References
1. Chen, Jiawen, Sylvain Paris, and Frédo Durand. "Real-time edge-aware image processing with the bilateral grid." ACM Transactions on Graphics (TOG) 26.3 (2007): 103-es.
