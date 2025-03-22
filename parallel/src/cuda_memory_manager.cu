#include "cuda_memory_manager.h"
#include <iostream>
#include <sstream>

namespace drumforge {

// Initialize the singleton instance pointer
CudaMemoryManager* CudaMemoryManager::instance = nullptr;

// Error checking helper implementation
void checkCudaError(cudaError_t error, const char* file, int line) {
    if (error != cudaSuccess) {
        std::stringstream ss;
        ss << "CUDA Error: " << cudaGetErrorString(error)
           << " at " << file << ":" << line;
        throw CudaException(ss.str());
    }
}

// Constructor
CudaMemoryManager::CudaMemoryManager() {
    // No initialization here - this is done in initialize() to allow better error handling
}

// Destructor
CudaMemoryManager::~CudaMemoryManager() {
    shutdown();
}

// Get singleton instance
CudaMemoryManager& CudaMemoryManager::getInstance() {
    if (instance == nullptr) {
        instance = new CudaMemoryManager();
    }
    return *instance;
}

// Initialize CUDA and check device capabilities
void CudaMemoryManager::initialize() {
    // Initialize CUDA runtime
    CUDA_CHECK(cudaSetDevice(0));
    
    // Print device information
    cudaDeviceProp props = getDeviceProperties();
    
    std::cout << "CUDA Device: " << props.name << std::endl;
    std::cout << "  Compute Capability: " << props.major << "." << props.minor << std::endl;
    std::cout << "  Global Memory: " << props.totalGlobalMem / (1024 * 1024) << " MB" << std::endl;
    std::cout << "  SM Count: " << props.multiProcessorCount << std::endl;
    std::cout << "  Max Threads per Block: " << props.maxThreadsPerBlock << std::endl;
    
    // Check for minimum compute capability (adjust as needed for your kernels)
    if (props.major < 3) {
        throw CudaException("This application requires a CUDA device with compute capability 3.0 or higher");
    }
}

// Shut down and release all resources
void CudaMemoryManager::shutdown() {
    // Clean up GL buffer resources
    for (auto buffer : registeredGLBuffers) {
        if (buffer != nullptr) {
            buffer->unregister();
        }
    }
    registeredGLBuffers.clear();
    
    // No need to manually free CudaBuffer objects as they're handled by shared_ptr
    allocatedBuffers.clear();
    
    // Reset device to clear all memory
    CUDA_CHECK(cudaDeviceReset());
}

// Register an OpenGL buffer for CUDA interop
std::shared_ptr<CudaGLBuffer> CudaMemoryManager::registerGLBuffer(GLuint buffer, unsigned int flags) {
    auto glBuffer = std::make_shared<CudaGLBuffer>();
    glBuffer->registerBuffer(buffer, flags);
    registeredGLBuffers.push_back(glBuffer.get());
    return glBuffer;
}

// Get device properties
cudaDeviceProp CudaMemoryManager::getDeviceProperties() {
    int deviceId;
    CUDA_CHECK(cudaGetDevice(&deviceId));
    
    cudaDeviceProp props;
    CUDA_CHECK(cudaGetDeviceProperties(&props, deviceId));
    
    return props;
}

} // namespace drumforge