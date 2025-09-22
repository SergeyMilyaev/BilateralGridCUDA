#include <iostream>

#include <Exceptions.h>
#include <ImageIO.h>
#include <ImagesCPU.h>
#include <ImagesNPP.h>

#include <helper_cuda.h>
#include <helper_string.h>

#include <npp.h>

#include "bilateralGrid.h"

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

    try
    {
        std::string sFilename;
        char *filePath;

        findCudaDevice(argc, (const char **)argv);

        if (printfNPPinfo(argc, argv) == false)
        {
            exit(EXIT_SUCCESS);
        }

        if (checkCmdLineFlag(argc, (const char **)argv, "input"))
        {
            getCmdLineArgumentString(argc, (const char **)argv, "input", &filePath);
        }
        else
        {
            filePath = sdkFindFilePath("Lena.pgm", argv[0]);
        }

        if (filePath)
        {
            sFilename = filePath;
        }
        else
        {
            sFilename = "Lena.pgm";
        }

        // if we specify the filename at the command line, then we only test
        // sFilename[0].
        int file_errors = 0;
        std::ifstream infile(sFilename.data(), std::ifstream::in);

        if (infile.good())
        {
            std::cout << "nppiRotate opened: <" << sFilename.data()
                      << "> successfully!" << std::endl;
            file_errors = 0;
            infile.close();
        }
        else
        {
            std::cout << "nppiRotate unable to open: <" << sFilename.data() << ">"
                      << std::endl;
            file_errors++;
            infile.close();
        }

        if (file_errors > 0)
        {
            exit(EXIT_FAILURE);
        }

        std::string sResultFilename = sFilename;

        std::string::size_type dot = sResultFilename.rfind('.');

        if (dot != std::string::npos)
        {
            sResultFilename = sResultFilename.substr(0, dot);
        }

        sResultFilename += "_filtered.pgm";

        if (checkCmdLineFlag(argc, (const char **)argv, "output"))
        {
            char *outputFilePath;
            getCmdLineArgumentString(argc, (const char **)argv, "output",
                                     &outputFilePath);
            sResultFilename = outputFilePath;
        }

        // declare a host image object for an 8-bit grayscale image
        npp::ImageCPU_8u_C1 oHostSrc;
        // load gray-scale image from disk
        npp::loadImage(sFilename, oHostSrc);
        // declare a device image and copy construct from the host image,
        // i.e. upload host to device
        npp::ImageNPP_8u_C1 oDeviceSrc(oHostSrc);

        // create device image for destination
        npp::ImageNPP_8u_C1 oDeviceDst(oDeviceSrc.size());

        // create struct with ROI size
        NppiSize oSizeROI = {(int)oDeviceSrc.width(), (int)oDeviceSrc.height()};

        // Bilateral filter parameters
        const int nMaskSize = 5;
        const float nValSquareSigma = 50.0f * 50.0f;
        const float nPosSquareSigma = 50.0f * 50.0f;
        const int nStepBetweenSrcPixels = 1;
        NppiSize oSrcSize = {(int)oDeviceSrc.width(), (int)oDeviceSrc.height()};
        NppiPoint oSrcOffset = {0, 0};

        // Create a CUDA stream
        cudaStream_t hStream;
        cudaStreamCreate(&hStream);

        // Create and populate the NppStreamContext
        NppStreamContext nppStreamCtx;
        nppStreamCtx.hStream = hStream;

        // perform bilateral filter
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start, nullptr);

        NPP_CHECK_NPP(nppiFilterBilateralGaussBorder_8u_C1R_Ctx(
            oDeviceSrc.data(), oDeviceSrc.pitch(), oSrcSize, oSrcOffset, 
            oDeviceDst.data(), oDeviceDst.pitch(), oSizeROI, nMaskSize, 
            nStepBetweenSrcPixels, nValSquareSigma, nPosSquareSigma, 
            NPP_BORDER_REPLICATE, nppStreamCtx));

        cudaEventRecord(stop, nullptr);
        cudaDeviceSynchronize();
        float start_stop;
        cudaEventElapsedTime(&start_stop, start, stop);
        std::cout << "Kernel execution time: " << start_stop / 1000 << " ms." << std::endl;

        // declare a host image for the result
        npp::ImageCPU_8u_C1 oHostDst(oDeviceDst.size());
        // and copy the device result data into it
        oDeviceDst.copyTo(oHostDst.data(), oHostDst.pitch());

        // save image to disk
        npp::saveImage(sResultFilename, oHostDst);
        std::cout << "Saved image: " << sResultFilename << std::endl;

        std::cout << "Input image width: " << oDeviceSrc.width() << " Input image pitch: " << oDeviceSrc.pitch() << std::endl;
        std::cout << "Result image width: " << oDeviceDst.width() << " Result image pitch: " << oDeviceDst.pitch() << std::endl;

        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start, nullptr);

        // Call our bilateral grid filter
        float scale_spatial = 16.0f; // spatial standard deviation
        float scale_range = 32.0f;   // range standard deviation
        bilateralGridFilter(oDeviceSrc.data(), oDeviceSrc.width(), oDeviceSrc.height(), oDeviceSrc.pitch(),
                            oDeviceDst.data(), oDeviceDst.pitch(), scale_spatial, scale_range);
        
        cudaEventRecord(stop, nullptr);
        cudaDeviceSynchronize();
        start_stop;
        cudaEventElapsedTime(&start_stop, start, stop);
        std::cout << "Kernel execution time: " << start_stop / 1000 << " ms." << std::endl;

        // and copy the device result data into it
        oDeviceDst.copyTo(oHostDst.data(), oHostDst.pitch());

        // save image to disk
        // npp::saveImage(sResultFilename, oHostDst);
        // std::cout << "Saved image: " << sResultFilename << std::endl;

        // free device images
        nppiFree(oDeviceSrc.data());
        nppiFree(oDeviceDst.data());

        cudaStreamDestroy(hStream);

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