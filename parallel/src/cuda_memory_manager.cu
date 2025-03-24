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
    std::cout << "Initializing CUDA..." << std::endl;
    
    // Check if CUDA devices are available
    int deviceCount;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    
    if (deviceCount == 0) {
        throw CudaException("No CUDA-capable devices found");
    }
    
    // Get the device with the highest compute capability
    // This is important for interop as newer devices tend to have better support
    int bestDevice = 0;
    int bestMajor = 0, bestMinor = 0;
    
    for (int dev = 0; dev < deviceCount; dev++) {
        cudaDeviceProp props;
        CUDA_CHECK(cudaGetDeviceProperties(&props, dev));
        
        // Check if this device has better compute capability
        if (props.major > bestMajor || 
            (props.major == bestMajor && props.minor > bestMinor)) {
            bestDevice = dev;
            bestMajor = props.major;
            bestMinor = props.minor;
        }
    }
    
    std::cout << "Selected CUDA device " << bestDevice << std::endl;
    
    // Set device flags before setting device
    // These flags are critical for proper CUDA-OpenGL interop
    CUDA_CHECK(cudaSetDeviceFlags(
        cudaDeviceScheduleAuto |      // Let CUDA decide on scheduling
        cudaDeviceMapHost |           // Enable mapped host memory
        cudaDeviceLmemResizeToMax     // Allocate maximum device memory
    ));
    
    // Set the device
    CUDA_CHECK(cudaSetDevice(bestDevice));
    
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
    
    // Check for canMapHostMemory capability (required for some interop operations)
    if (!props.canMapHostMemory) {
        std::cout << "Warning: Device does not support mapping host memory. This may affect performance." << std::endl;
    }
    
    // Check if OpenGL interop is supported on this device
    bool glInteropSupported = isGLInteropSupported();
    std::cout << "CUDA-OpenGL Interoperability: " 
              << (glInteropSupported ? "Supported" : "Not Supported") << std::endl;
    
    if (!glInteropSupported) {
        throw CudaException("CUDA-OpenGL interoperability is not supported on this device. "
                            "The application requires interop capabilities to run.");
    }
    
    // Initialize CUDA runtime by performing a small allocation
    // This ensures CUDA is fully initialized before any interop operations
    void* testPtr = nullptr;
    CUDA_CHECK(cudaMalloc(&testPtr, 1));
    CUDA_CHECK(cudaFree(testPtr));
    
    std::cout << "CUDA initialized successfully" << std::endl;
}

// Shut down and release all resources
void CudaMemoryManager::shutdown() {
    // No need to manually free CudaBuffer objects as they're handled by shared_ptr
    allocatedBuffers.clear();
    
    std::cout << "Shutting down CUDA..." << std::endl;
    
    // Synchronize device before reset
    try {
        synchronize();
    } 
    catch (const CudaException& e) {
        std::cerr << "Warning during shutdown synchronization: " << e.what() << std::endl;
    }
    
    // Reset device to clear all memory
    try {
        CUDA_CHECK(cudaDeviceReset());
    }
    catch (const CudaException& e) {
        std::cerr << "Warning during device reset: " << e.what() << std::endl;
    }
    
    std::cout << "CUDA shutdown complete" << std::endl;
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
        // Ensure CUDA device is synchronized before registering
        synchronize();
        
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
        // Ensure CUDA device is synchronized before creating buffer
        synchronize();
        
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

// NOTE: Removed duplicate synchronize() method - it's already defined in the header file

} // namespace drumforge