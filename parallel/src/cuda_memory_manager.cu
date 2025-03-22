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
    
    // Check if OpenGL interop is supported
    bool glInteropSupported = isGLInteropSupported();
    std::cout << "CUDA-OpenGL Interoperability: " 
              << (glInteropSupported ? "Supported" : "Not Supported") << std::endl;
              
    if (!glInteropSupported) {
        std::cout << "Warning: CUDA-OpenGL interoperability is not supported on this device. "
                  << "Visualization features may not be available." << std::endl;
    }
}

// Shut down and release all resources
void CudaMemoryManager::shutdown() {
    // No need to manually free CudaBuffer objects as they're handled by shared_ptr
    allocatedBuffers.clear();
    
    // Reset device to clear all memory
    CUDA_CHECK(cudaDeviceReset());
}

// Get device properties
cudaDeviceProp CudaMemoryManager::getDeviceProperties() {
    int deviceId;
    CUDA_CHECK(cudaGetDevice(&deviceId));
    
    cudaDeviceProp props;
    CUDA_CHECK(cudaGetDeviceProperties(&props, deviceId));
    
    return props;
}

// Check if CUDA-OpenGL interop is supported on this device
bool CudaMemoryManager::isGLInteropSupported() {
    return cuda_gl::isInteropSupported();
}

// Register an existing OpenGL buffer with CUDA
std::shared_ptr<CudaGLBuffer> CudaMemoryManager::registerGLBuffer(
    GLuint bufferId, size_t count, size_t elemSize, 
    cudaGraphicsRegisterFlags flags) {
    
    try {
        // Use the cuda_gl helper to register the buffer
        auto buffer = cuda_gl::registerBuffer(bufferId, count, elemSize, flags);
        std::cout << "Registered OpenGL buffer " << bufferId << " with CUDA" << std::endl;
        return buffer;
    }
    catch (const CudaException& e) {
        std::cerr << "Failed to register OpenGL buffer: " << e.what() << std::endl;
        throw; // Re-throw to let caller handle it
    }
}

// Create a new OpenGL buffer and register it with CUDA
std::shared_ptr<CudaGLBuffer> CudaMemoryManager::createGLBuffer(
    size_t count, size_t elemSize, GLenum target, GLenum usage) {
    
    try {
        // Use the cuda_gl helper to create and register a new buffer
        auto buffer = cuda_gl::createBuffer(count, elemSize, target, usage);
        std::cout << "Created new OpenGL buffer and registered with CUDA" 
                  << " (size: " << count << " elements, " 
                  << (count * elemSize) << " bytes)" << std::endl;
        return buffer;
    }
    catch (const CudaException& e) {
        std::cerr << "Failed to create and register OpenGL buffer: " << e.what() << std::endl;
        throw; // Re-throw to let caller handle it
    }
}

} // namespace drumforge