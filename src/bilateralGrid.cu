#include "bilateralGrid.h"
#include <cstdio>

__global__ void splat_kernel(const unsigned char* inp_img, int inp_img_width, int inp_img_height, int inp_img_pitch,
                float* grid_value_sum, float* grid_weight_sum, int grid_width, int grid_height, int grid_depth,
                float scale_spatial, float scale_range)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= inp_img_width || y >= inp_img_height) 
        return;

    float inp_val = (float) inp_img[y * inp_img_pitch + x];

    // compute grid coords
    int grid_x = (int) roundf((float) x / scale_spatial);
    int grid_y = (int) roundf((float) y / scale_spatial);
    int grid_r = (int) roundf(inp_val / scale_range);

    // clamp
    grid_x = min(max(grid_x,0), grid_width-1);
    grid_y = min(max(grid_y,0), grid_height-1);
    grid_r = min(max(grid_r,0), grid_depth-1);

    // grid index
    int idx = (grid_r * grid_height + grid_y) * grid_width + grid_x;

    // atomic add
    atomicAdd(&grid_value_sum[idx], inp_val);
    atomicAdd(&grid_weight_sum[idx], 1.0f);
}

__global__
void slice_kernel(const unsigned char* inp_img, int inp_img_width, int inp_img_height, int inp_img_pitch, 
                unsigned char* out_img, int out_img_pitch, const float* grid_value_sum, const float* grid_weight_sum,
                int grid_width, int grid_height, int grid_depth, float scale_spatial, float scale_range)
{
    // Function implementation
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= inp_img_width || y >= inp_img_height) return;

    float val = (float) inp_img[y * inp_img_pitch + x];
    // continuous grid coordinate
    float grid_x_f = (float) x / (float) scale_spatial;
    float grid_y_f = (float) y / (float) scale_spatial;
    float grid_r_f = (float) val / (float) scale_range;

    // int gx = (int) roundf(grid_x_f);
    // int gy = (int) roundf(grid_y_f);
    // int gr = (int) roundf(grid_r_f);

    // gx = min(max(gx,0), grid_width-1);
    // gy = min(max(gy,0), grid_height-1);
    // gr = min(max(gr,0), grid_depth-1);

    // float vsum_interp0 = grid_value_sum[(gr*grid_height+gy)*grid_width+gx];
    // float wsum_interp0 = grid_weight_sum[(gr*grid_height+gy)*grid_width+gx];

    // out_img[y * out_img_pitch + x] = (unsigned char) min(max(vsum_interp0 / max(wsum_interp0, 1.f), 0.f), 255.f);
    // return;

    // do trilinear interpolation around (grid_x_f, grid_y_f, grid_r_f)
    int gx0 = (int) floorf(grid_x_f);
    int gy0 = (int) floorf(grid_y_f);
    int gr0 = (int) floorf(grid_r_f);
    int gx1 = gx0 + 1;
    int gy1 = gy0 + 1;
    int gr1 = gr0 + 1;

    gx0 = min(max(gx0, 0), grid_width-1);
    gx1 = min(max(gx1, 0), grid_width-1);
    gy0 = min(max(gy0, 0), grid_height-1);
    gy1 = min(max(gy1, 0), grid_height-1);
    gr0 = min(max(gr0, 0), grid_depth-1);
    gr1 = min(max(gr1, 0), grid_depth-1);

    // interpolation weights
    float dx = grid_x_f - (float) gx0;
    float dy = grid_y_f - (float) gy0;
    float dr = grid_r_f - (float) gr0;

    // float vsum_interp = 0.0f;
    // float wsum_interp = 0.0f;

    auto v000 = grid_value_sum[(gr0*grid_height+gy0)*grid_width+gx0];
    auto v001 = grid_value_sum[(gr0*grid_height+gy0)*grid_width+gx1];
    auto v010 = grid_value_sum[(gr0*grid_height+gy1)*grid_width+gx0];
    auto v011 = grid_value_sum[(gr0*grid_height+gy1)*grid_width+gx1];
    auto v100 = grid_value_sum[(gr1*grid_height+gy0)*grid_width+gx0];
    auto v101 = grid_value_sum[(gr1*grid_height+gy0)*grid_width+gx1];
    auto v110 = grid_value_sum[(gr1*grid_height+gy1)*grid_width+gx0];
    auto v111 = grid_value_sum[(gr1*grid_height+gy1)*grid_width+gx1];

    auto w000 = grid_weight_sum[(gr0*grid_height+gy0)*grid_width+gx0];
    auto w001 = grid_weight_sum[(gr0*grid_height+gy0)*grid_width+gx1];
    auto w010 = grid_weight_sum[(gr0*grid_height+gy1)*grid_width+gx0];
    auto w011 = grid_weight_sum[(gr0*grid_height+gy1)*grid_width+gx1];
    auto w100 = grid_weight_sum[(gr1*grid_height+gy0)*grid_width+gx0];
    auto w101 = grid_weight_sum[(gr1*grid_height+gy0)*grid_width+gx1];
    auto w110 = grid_weight_sum[(gr1*grid_height+gy1)*grid_width+gx0];
    auto w111 = grid_weight_sum[(gr1*grid_height+gy1)*grid_width+gx1];

    // trilinear interp sums
    float c00 = v000*(1.0f-dx) + v001*dx;
    float c01 = v010*(1.0f-dx) + v011*dx;
    float c10 = v100*(1.0f-dx) + v101*dx;
    float c11 = v110*(1.0f-dx) + v111*dx;
    float c0 = c00*(1.0f-dy) + c01*dy;
    float c1 = c10*(1.0f-dy) + c11*dy;
    float vsum_interp = c0*(1.0f-dr) + c1*dr;

    // trilinear interp weights
    c00 = w000*(1.0f-dx) + w001*dx;
    c01 = w010*(1.0f-dx) + w011*dx;
    c10 = w100*(1.0f-dx) + w101*dx;
    c11 = w110*(1.0f-dx) + w111*dx;
    c0 = c00*(1.0f-dy) + c01*dy;
    c1 = c10*(1.0f-dy) + c11*dy;
    float wsum_interp = c0*(1.0f-dr) + c1*dr;

    // for (int i = 0; i < 2; ++i) {
    //     for (int j = 0; j < 2; ++j) {
    //         for (int k = 0; k < 2; ++k) {
    //             int gx = gx0 + i;
    //             int gy = gy0 + j;
    //             int gr = gr0 + k;

    //             // clamp coordinates
    //             gx = min(max(gx, 0), grid_width - 1);
    //             gy = min(max(gy, 0), grid_height - 1);
    //             gr = min(max(gr, 0), grid_depth - 1);

    //             int idx = (gr * grid_height + gy) * grid_width + gx;
                
    //             float v = grid_value_sum[idx];
    //             float w = grid_weight_sum[idx];

    //             float interp_weight = (i == 0 ? 1 - wx : wx) *
    //                                   (j == 0 ? 1 - wy : wy) *
    //                                   (k == 0 ? 1 - wr : wr);
                
    //             vsum_interp += v * interp_weight;
    //             wsum_interp += w * interp_weight;
    //         }
    //     }
    // }

    out_img[y * out_img_pitch + x] = (unsigned char) min(max(vsum_interp / max(wsum_interp, 1.f), 0.f), 255.f);
}


__host__ void bilateralGridFilter(const unsigned char* d_input, int inp_img_width, int inp_img_height, int inp_img_pitch,
                        unsigned char* d_output, int out_img_pitch, float scale_spatial, float scale_range)
{
    // Define grid dimensions
    int grid_width = (inp_img_width + scale_spatial - 1) / scale_spatial;
    int grid_height = (inp_img_height + scale_spatial - 1) / scale_spatial;
    int grid_depth = (256 + scale_range - 1) / scale_range;
    printf("Grid dimensions: %d x %d x %d\n", grid_width, grid_height, grid_depth);

    // Allocate grid memory
    size_t grid_size = grid_width * grid_height * grid_depth * sizeof(float);
    float* d_grid_value_sum;
    float* d_grid_weight_sum;
    cudaMalloc(&d_grid_value_sum, grid_size);
    cudaMalloc(&d_grid_weight_sum, grid_size);
    cudaMemset(d_grid_value_sum, 0, grid_size);
    cudaMemset(d_grid_weight_sum, 0, grid_size);

    // Launch splat kernel
    dim3 blockSize(16, 16);
    dim3 gridSize((inp_img_width + blockSize.x - 1) / blockSize.x,
                  (inp_img_height + blockSize.y - 1) / blockSize.y);
    splat_kernel<<<gridSize, blockSize>>>(d_input, inp_img_width, inp_img_height, inp_img_pitch,
                                        d_grid_value_sum, d_grid_weight_sum, grid_width, grid_height, grid_depth,
                                        scale_spatial, scale_range);
    cudaDeviceSynchronize();

    // for (int i = 0; i < grid_size / sizeof(float); i += grid_width * grid_height) {
    //     float val_sum, weight_sum;
    //     cudaMemcpy(&val_sum, &d_grid_value_sum[i], sizeof(float), cudaMemcpyDeviceToHost);
    //     cudaMemcpy(&weight_sum, &d_grid_weight_sum[i], sizeof(float), cudaMemcpyDeviceToHost);
    //     printf("Grid cell %d: value sum = %f, weight sum = %f\n", i / (grid_width * grid_height), val_sum, weight_sum);
    // }

    // Launch slice kernel
    slice_kernel<<<gridSize, blockSize>>>(d_input, inp_img_width, inp_img_height, inp_img_pitch,
                                        d_output, out_img_pitch, 
                                        d_grid_value_sum, d_grid_weight_sum, grid_width, grid_height, grid_depth,
                                        scale_spatial, scale_range);
    cudaDeviceSynchronize();

    // for (int i = 0; i < 3 * scale_spatial; ++i) 
    // {
    //     for (int j = 0; j < 3 * scale_spatial; ++j) 
    //     {
    //         unsigned char pixel;
    //         cudaMemcpy(&pixel, &d_output[i * out_img_pitch + j], sizeof(unsigned char), cudaMemcpyDeviceToHost);
    //         printf("%3d ", pixel);
    //     }
    //     printf("\n");
    // }

    // Free grid memory
    cudaFree(d_grid_value_sum);
    cudaFree(d_grid_weight_sum);
}