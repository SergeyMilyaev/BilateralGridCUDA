#include <iostream>

#include <Exceptions.h>
#include <ImageIO.h>
#include <ImagesCPU.h>
#include <ImagesNPP.h>

#include <helper_cuda.h>
#include <helper_string.h>

#include <npp.h>

#include "bilateralGrid.h"

#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <cstring> // added for C-string handling

bool printfNPPinfo(int argc, char *argv[])
{
  const NppLibraryVersion *libVer = nppGetLibVersion();

  printf("NPP Library Version %d.%d.%d\n", libVer->major, libVer->minor,
         libVer->build);

  int driverVersion, runtimeVersion;
  cudaDriverGetVersion(&driverVersion);
  cudaRuntimeGetVersion(&runtimeVersion);

  printf("  CUDA Driver  Version: %d.%d\n", driverVersion / 1000,
         (driverVersion % 100) / 10);
  printf("  CUDA Runtime Version: %d.%d\n", runtimeVersion / 1000,
         (runtimeVersion % 100) / 10);

  // Min spec is SM 1.0 devices
  bool bVal = checkCudaCapabilities(1, 0);
  return bVal;
}

int main(int argc, char *argv[])
{
    printf("%s Starting...\n\n", argv[0]);

    if (checkCmdLineFlag(argc, (const char **)argv, "help"))
    {
        printf("Usage: %s [options]\n", argv[0]);
        printf("Options:\n");
        printf("  -filter_radius=<int>        NPP bilateral filter mask size (default: 5)\n");
        printf("  -sigma_v=<float>        NPP bilateral filter value sigma (default: 50.0)\n");
        printf("  -sigma_p=<float>        NPP bilateral filter position sigma (default: 50.0)\n");
        printf("  -grid_sigma_s=<float>   Bilateral grid spatial sigma (default: 16.0)\n");
        printf("  -grid_sigma_r=<float>   Bilateral grid range sigma (default: 32.0)\n");
        printf("  -filelist=<path>        Path to a custom file list (default: data/img_list_attribution.txt)\n");
        printf("  -input_dir=<dir>        Directory containing input images (default: data/input/)\n");
        printf("  -output_dir=<dir>       Directory to write output images (default: data/output/)\n");
        exit(EXIT_SUCCESS);
    }

    try
    {
        findCudaDevice(argc, (const char **)argv);

        if (printfNPPinfo(argc, argv) == false)
        {
            exit(EXIT_SUCCESS);
        }

        // Filter parameters
        int nRadius = 5;
        float nSigmaV = 50.0f;
        float nSigmaP = 50.0f;
        float grid_scale_spatial = 16.0f;
        float grid_scale_range = 32.0f;

        getCmdLineArgumentValue(argc, (const char **)argv, "filter_radius", &nRadius);
        getCmdLineArgumentValue(argc, (const char **)argv, "sigma_v", &nSigmaV);
        getCmdLineArgumentValue(argc, (const char **)argv, "sigma_p", &nSigmaP);
        getCmdLineArgumentValue(argc, (const char **)argv, "grid_sigma_s", &grid_scale_spatial);
        getCmdLineArgumentValue(argc, (const char **)argv, "grid_sigma_r", &grid_scale_range);

        // Default paths (can be overridden via command line)
        std::string list_file_path = "data/img_list_attribution.txt";
        std::string input_dir = "data/input/";
        std::string output_dir = "data/output/";

        // Simple manual parsing for -filelist=, -input_dir= and -output_dir=
        for (int i = 1; i < argc; ++i)
        {
            std::string arg = argv[i];
            const std::string fl_prefix = "-filelist=";
            const std::string in_prefix = "-input_dir=";
            const std::string out_prefix = "-output_dir=";
            if (arg.rfind(fl_prefix, 0) == 0)
            {
                list_file_path = arg.substr(fl_prefix.size());
            }
            else if (arg.rfind(in_prefix, 0) == 0)
            {
                input_dir = arg.substr(in_prefix.size());
            }
            else if (arg.rfind(out_prefix, 0) == 0)
            {
                output_dir = arg.substr(out_prefix.size());
            }
        }

        // Ensure directories end with a slash
        if (!input_dir.empty() && input_dir.back() != '/')
            input_dir.push_back('/');
        if (!output_dir.empty() && output_dir.back() != '/')
            output_dir.push_back('/');

        std::cout << "Using file list: " << list_file_path << std::endl;
        std::cout << "Input directory: " << input_dir << std::endl;
        std::cout << "Output directory: " << output_dir << std::endl;

        std::ifstream list_file(list_file_path);
        if (!list_file.is_open())
        {
            std::cerr << "Error: Unable to open image list file " << list_file_path << std::endl;
            exit(EXIT_FAILURE);
        }

        std::string line;
        while (std::getline(list_file, line))
        {
            std::stringstream ss(line);
            std::string image_name;
            ss >> image_name;

            if (image_name.empty()) continue;

            // Use provided input_dir
            std::string sFilename = input_dir + image_name;

            std::cout << "Processing: " << sFilename << std::endl;

            // Try load an image
            npp::ImageCPU_8u_C1 oHostSrc;            
            try
            {
                npp::loadImage(sFilename, oHostSrc);
            }
            catch (const npp::Exception &e)
            {
                std::cerr << "Failed to load image " << sFilename << ". Error: " << e << std::endl;
                continue; // Skip to next image
            }
            
            npp::ImageNPP_8u_C1 oDeviceSrc(oHostSrc);  //a device image and copy construct from the host image,
            npp::ImageNPP_8u_C1 oDeviceDst(oDeviceSrc.size());  // create device image for destination
            npp::ImageCPU_8u_C1 oHostDst(oDeviceDst.size()); // declare a host image for the result

           
            // Create a CUDA stream
            cudaStream_t hStream;
            cudaStreamCreate(&hStream);

            // Create and populate the NppStreamContext
            NppStreamContext nppStreamCtx;
            nppStreamCtx.hStream = hStream;

            // --- Process with nppiFilterBilateralGaussBorder ---
            
            cudaEvent_t start, stop;
            cudaEventCreate(&start);
            cudaEventCreate(&stop);
            cudaEventRecord(start, nullptr);

            // Bilateral filter parameters
            const float nValSquareSigma = nSigmaV * nSigmaV;
            const float nPosSquareSigma = nSigmaP * nSigmaP;
            const int nStepBetweenSrcPixels = 1;
            NppiSize oSrcSize = {(int)oDeviceSrc.width(), (int)oDeviceSrc.height()};
            NppiPoint oSrcOffset = {0, 0};
            NppiSize oSizeROI = {(int)oDeviceSrc.width(), (int)oDeviceSrc.height()};

            NPP_CHECK_NPP(nppiFilterBilateralGaussBorder_8u_C1R_Ctx(
                oDeviceSrc.data(), oDeviceSrc.pitch(), oSrcSize, oSrcOffset, 
                oDeviceDst.data(), oDeviceDst.pitch(), oSizeROI, nRadius, 
                nStepBetweenSrcPixels, nValSquareSigma, nPosSquareSigma, 
                NPP_BORDER_REPLICATE, nppStreamCtx));
            cudaDeviceSynchronize();

            cudaEventRecord(stop, nullptr);
            cudaDeviceSynchronize();
            float start_stop;
            cudaEventElapsedTime(&start_stop, start, stop);
            std::cout << "NPP bilateral filter execution time: " << start_stop / 1000 << " ms." << std::endl;
            oDeviceDst.copyTo(oHostDst.data(), oHostDst.pitch());

            // Remove original extension from the filename before adding result suffix
            std::string base_name = image_name;
            size_t last_dot = base_name.find_last_of('.');
            size_t last_slash = base_name.find_last_of("/\\");
            if (last_dot != std::string::npos &&
                (last_slash == std::string::npos || last_dot > last_slash))
            {
                base_name = base_name.substr(0, last_dot);
            }

            // Use provided output_dir and base_name (no original extension)
            std::string sResultFilename = output_dir + base_name + "_npp_bilateral_filtered.pgm";
            npp::saveImage(sResultFilename, oHostDst);
            std::cout << "  Saved NPP filtered image: " << sResultFilename << std::endl;
            

            // --- Process with bilateralGridFilter ---
            
            cudaEventCreate(&start);
            cudaEventCreate(&stop);
            cudaEventRecord(start, nullptr);

            bilateralGridFilter(oDeviceSrc.data(), oDeviceSrc.width(), oDeviceSrc.height(), oDeviceSrc.pitch(),
                                oDeviceDst.data(), oDeviceDst.pitch(), grid_scale_spatial, grid_scale_range);
            cudaDeviceSynchronize();
            
            cudaEventRecord(stop, nullptr);
            cudaDeviceSynchronize();
            cudaEventElapsedTime(&start_stop, start, stop);
            std::cout << "Bilateral grid filter execution time: " << start_stop / 1000 << " ms." << std::endl;

            oDeviceDst.copyTo(oHostDst.data(), oHostDst.pitch());

            sResultFilename = output_dir + base_name + "_bilateral_grid_filtered.pgm";
            npp::saveImage(sResultFilename, oHostDst);
            std::cout << "  Saved grid filtered image: " << sResultFilename << std::endl;
            

            nppiFree(oDeviceSrc.data());
            nppiFree(oDeviceDst.data());

            cudaStreamDestroy(hStream);
        }

        list_file.close();
        exit(EXIT_SUCCESS);
    }
    catch (npp::Exception &rException)
    {
        std::cerr << "Program error! The following exception occurred: \n";
        std::cerr << rException << std::endl;
        std::cerr << "Aborting." << std::endl;

        exit(EXIT_FAILURE);
    }
    catch (...)
    {
        std::cerr << "Program error! An unknown type of exception occurred. \n";
        std::cerr << "Aborting." << std::endl;

        exit(EXIT_FAILURE);
        return -1;
    }

    return 0;
}