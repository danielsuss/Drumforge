#include <iostream>
#include <cuda_runtime.h>

// Simple CUDA kernel that does nothing but prove CUDA works
__global__ void helloKernel() {
    // Empty kernel, just to test CUDA compilation and execution
}

// Function declaration for external linkage
extern "C" bool testCuda();

// Function to test basic CUDA functionality
bool testCuda() {
    // Print CUDA device properties
    int deviceCount = 0;
    cudaError_t error = cudaGetDeviceCount(&deviceCount);
    
    if (error != cudaSuccess) {
        std::cerr << "Error getting CUDA device count: " 
                  << cudaGetErrorString(error) << std::endl;
        return false;
    }
    
    std::cout << "Found " << deviceCount << " CUDA device(s)" << std::endl;
    
    // Get properties for each device
    for (int i = 0; i < deviceCount; ++i) {
        cudaDeviceProp prop;
        if (cudaGetDeviceProperties(&prop, i) == cudaSuccess) {
            std::cout << "Device " << i << ": " << prop.name << std::endl;
            std::cout << "  Compute capability: " << prop.major << "." << prop.minor << std::endl;
            std::cout << "  Total global memory: " << prop.totalGlobalMem / (1024 * 1024) << " MB" << std::endl;
            std::cout << "  Max threads per block: " << prop.maxThreadsPerBlock << std::endl;
        }
    }
    
    // Launch an empty kernel just to verify CUDA execution works
    helloKernel<<<1, 1>>>();
    
    // Check for kernel launch errors
    cudaError_t kernelError = cudaGetLastError();
    if (kernelError != cudaSuccess) {
        std::cerr << "Kernel launch failed: " << cudaGetErrorString(kernelError) << std::endl;
        return false;
    }
    
    // Wait for GPU to finish
    if (cudaDeviceSynchronize() != cudaSuccess) {
        std::cerr << "CUDA synchronize failed: " << cudaGetErrorString(cudaGetLastError()) << std::endl;
        return false;
    }
    
    std::cout << "CUDA kernel executed successfully!" << std::endl;
    return true;
}