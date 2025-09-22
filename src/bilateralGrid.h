#pragma once

#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
#define WINDOWS_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#pragma warning(disable : 4819)
#endif

#include <cuda_runtime.h>


__global__ void splat_kernel(const unsigned char* inp_img, int inp_img_width, int inp_img_height, int inp_img_pitch,
                float* grid_value_sum, float* grid_weight_sum, int grid_width, int grid_height, int grid_depth,
                float scale_spatial, float scale_range);

__global__ void slice_kernel(const unsigned char* inp_img, int inp_img_width, int inp_img_height, int inp_img_pitch, 
                unsigned char* out_img, int out_img_pitch, const float* grid_value_sum, const float* grid_weight_sum,
                int grid_width, int grid_height, int grid_depth, float scale_spatial, float scale_range);

__host__ void bilateralGridFilter(const unsigned char* d_input, int inp_img_width, int inp_img_height, int inp_img_pitch,
                        unsigned char* d_output, int out_img_pitch, float scale_spatial, float scale_range);