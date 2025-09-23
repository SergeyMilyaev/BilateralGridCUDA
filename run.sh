#!/bin/bash

# Defaults (match defaults in src/main.cpp)
FILELIST="data/img_list_attribution.txt"
INPUT_DIR="data/input/"
OUTPUT_DIR="data/output/"

# Run the program with explicit input/output/filelist args
./bin/bilateralGrid -filter_radius=32 -filelist="${FILELIST}" -input_dir="${INPUT_DIR}" -output_dir="${OUTPUT_DIR}"