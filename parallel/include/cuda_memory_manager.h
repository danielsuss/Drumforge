#ifndef DRUMFORGE_CUDA_MEMORY_MANAGER_H
#define DRUMFORGE_CUDA_MEMORY_MANAGER_H

// Add these preprocessor definitions before any includes
#ifdef _WIN32
#define NOMINMAX // Prevent Windows from defining min/max macros
#endif

// Include OpenGL headers first
#include <GL/glew.h>

// Then include CUDA headers
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

#include <vector>
#include <string>
#include <memory>
#include <stdexcept>

namespace drumforge {

// Simple exception class for CUDA errors
class CudaException : public std::runtime_error {
public:
    CudaException(const std::string& message) : std::runtime_error(message) {}
};

// Helper function to check CUDA errors
void checkCudaError(cudaError_t error, const char* file, int line);

// Macro for easier error checking
#define CUDA_CHECK(call) checkCudaError(call, __FILE__, __LINE__)

// Template class to wrap a CUDA device buffer
template<typename T>
class CudaBuffer {
private:
    T* devicePtr;
    size_t elementCount;
    size_t byteSize;
    
public:
    CudaBuffer() : devicePtr(nullptr), elementCount(0), byteSize(0) {}
    
    CudaBuffer(size_t count) : devicePtr(nullptr), elementCount(count), 
                            byteSize(count * sizeof(T)) {
        CUDA_CHECK(cudaMalloc(&devicePtr, byteSize));
    }
    
    ~CudaBuffer() {
        release();
    }
    
    // Prevent copying
    CudaBuffer(const CudaBuffer&) = delete;
    CudaBuffer& operator=(const CudaBuffer&) = delete;
    
    // Allow moving
    CudaBuffer(CudaBuffer&& other) noexcept 
        : devicePtr(other.devicePtr), elementCount(other.elementCount), 
          byteSize(other.byteSize) {
        other.devicePtr = nullptr;
        other.elementCount = 0;
        other.byteSize = 0;
    }
    
    CudaBuffer& operator=(CudaBuffer&& other) noexcept {
        if (this != &other) {
            release();
            devicePtr = other.devicePtr;
            elementCount = other.elementCount;
            byteSize = other.byteSize;
            other.devicePtr = nullptr;
            other.elementCount = 0;
            other.byteSize = 0;
        }
        return *this;
    }
    
    // Allocate memory
    void allocate(size_t count) {
        release();
        elementCount = count;
        byteSize = count * sizeof(T);
        CUDA_CHECK(cudaMalloc(&devicePtr, byteSize));
    }
    
    // Release memory
    void release() {
        if (devicePtr != nullptr) {
            CUDA_CHECK(cudaFree(devicePtr));
            devicePtr = nullptr;
        }
        elementCount = 0;
        byteSize = 0;
    }
    
    // Copy data from host to device
    void copyFromHost(const T* hostData) {
        if (devicePtr == nullptr || elementCount == 0) {
            throw CudaException("Cannot copy to unallocated buffer");
        }
        CUDA_CHECK(cudaMemcpy(devicePtr, hostData, byteSize, cudaMemcpyHostToDevice));
    }
    
    // Copy data from device to host
    void copyToHost(T* hostData) const {
        if (devicePtr == nullptr || elementCount == 0) {
            throw CudaException("Cannot copy from unallocated buffer");
        }
        CUDA_CHECK(cudaMemcpy(hostData, devicePtr, byteSize, cudaMemcpyDeviceToHost));
    }
    
    // Zero out the buffer
    void zero() {
        if (devicePtr == nullptr || elementCount == 0) {
            throw CudaException("Cannot zero unallocated buffer");
        }
        CUDA_CHECK(cudaMemset(devicePtr, 0, byteSize));
    }
    
    // Getters
    T* get() const { return devicePtr; }
    size_t size() const { return elementCount; }
    size_t bytes() const { return byteSize; }
};

// New class for CUDA-OpenGL interoperable buffers
template<typename T>
class CudaGLBuffer {
private:
    GLuint glBufferId;                    // OpenGL buffer ID
    cudaGraphicsResource_t cudaResource;  // CUDA graphics resource
    T* devicePtr;                         // Device pointer when mapped
    size_t elementCount;                  // Number of elements
    size_t byteSize;                      // Size in bytes
    bool isMapped;                        // Whether buffer is currently mapped

public:
    // Constructor and destructor
    CudaGLBuffer()
        : glBufferId(0), cudaResource(nullptr), devicePtr(nullptr),
          elementCount(0), byteSize(0), isMapped(false) {}
    
    CudaGLBuffer(GLuint bufferId, size_t count, cudaGraphicsRegisterFlags flags = cudaGraphicsRegisterFlagsNone)
        : glBufferId(0), cudaResource(nullptr), devicePtr(nullptr),
          elementCount(0), byteSize(0), isMapped(false) {
        registerBuffer(bufferId, count, flags);
    }
    
    ~CudaGLBuffer() {
        unregisterBuffer();
    }
    
    // Prevent copying
    CudaGLBuffer(const CudaGLBuffer&) = delete;
    CudaGLBuffer& operator=(const CudaGLBuffer&) = delete;
    
    // Allow moving
    CudaGLBuffer(CudaGLBuffer&& other) noexcept 
        : glBufferId(other.glBufferId), cudaResource(other.cudaResource),
          devicePtr(other.devicePtr), elementCount(other.elementCount),
          byteSize(other.byteSize), isMapped(other.isMapped) {
        other.glBufferId = 0;
        other.cudaResource = nullptr;
        other.devicePtr = nullptr;
        other.elementCount = 0;
        other.byteSize = 0;
        other.isMapped = false;
    }
    
    CudaGLBuffer& operator=(CudaGLBuffer&& other) noexcept {
        if (this != &other) {
            unregisterBuffer();
            glBufferId = other.glBufferId;
            cudaResource = other.cudaResource;
            devicePtr = other.devicePtr;
            elementCount = other.elementCount;
            byteSize = other.byteSize;
            isMapped = other.isMapped;
            other.glBufferId = 0;
            other.cudaResource = nullptr;
            other.devicePtr = nullptr;
            other.elementCount = 0;
            other.byteSize = 0;
            other.isMapped = false;
        }
        return *this;
    }
    
    // Register an OpenGL buffer with CUDA
    void registerBuffer(GLuint bufferId, size_t count, cudaGraphicsRegisterFlags flags = cudaGraphicsRegisterFlagsNone) {
        unregisterBuffer();
        
        glBufferId = bufferId;
        elementCount = count;
        byteSize = count * sizeof(T);
        
        CUDA_CHECK(cudaGraphicsGLRegisterBuffer(&cudaResource, glBufferId, flags));
    }
    
    // Unregister the buffer
    void unregisterBuffer() {
        if (isMapped) {
            unmap();
        }
        
        if (cudaResource != nullptr) {
            CUDA_CHECK(cudaGraphicsUnregisterResource(cudaResource));
            cudaResource = nullptr;
        }
        
        glBufferId = 0;
        elementCount = 0;
        byteSize = 0;
    }
    
    // Map the buffer for CUDA access
    T* map() {
        if (!cudaResource) {
            throw CudaException("Cannot map unregistered buffer");
        }
        
        if (!isMapped) {
            CUDA_CHECK(cudaGraphicsMapResources(1, &cudaResource));
            size_t size;
            CUDA_CHECK(cudaGraphicsResourceGetMappedPointer((void**)&devicePtr, &size, cudaResource));
            isMapped = true;
        }
        
        return devicePtr;
    }
    
    // Unmap the buffer
    void unmap() {
        if (cudaResource && isMapped) {
            CUDA_CHECK(cudaGraphicsUnmapResources(1, &cudaResource));
            isMapped = false;
            devicePtr = nullptr;
        }
    }
    
    // Getters
    T* getDevicePtr() const { 
        if (!isMapped) {
            throw CudaException("Buffer not mapped, cannot access device pointer");
        }
        return devicePtr; 
    }
    
    GLuint getGLBufferId() const { return glBufferId; }
    size_t size() const { return elementCount; }
    size_t bytes() const { return byteSize; }
    bool isRegistered() const { return cudaResource != nullptr; }
    bool isMappedForCUDA() const { return isMapped; }
};

// Main memory manager class
class CudaMemoryManager {
private:
    // Track all allocated buffers for potential cleanup
    std::vector<void*> allocatedBuffers;
    
    // Singleton instance
    static CudaMemoryManager* instance;
    
    // Private constructor for singleton
    CudaMemoryManager();
    
public:
    // No copying or assignment
    CudaMemoryManager(const CudaMemoryManager&) = delete;
    CudaMemoryManager& operator=(const CudaMemoryManager&) = delete;
    
    ~CudaMemoryManager();
    
    // Get singleton instance
    static CudaMemoryManager& getInstance();
    
    // Initialize CUDA and check device capabilities
    void initialize();
    
    // Shut down and release all resources
    void shutdown();
    
    // Allocate device memory
    template<typename T>
    std::shared_ptr<CudaBuffer<T>> allocateBuffer(size_t elementCount) {
        auto buffer = std::make_shared<CudaBuffer<T>>(elementCount);
        allocatedBuffers.push_back(buffer.get());
        return buffer;
    }
    
    // Create and manage a CUDA-GL interop buffer
    template<typename T>
    std::shared_ptr<CudaGLBuffer<T>> registerGLBuffer(GLuint bufferId, size_t count, 
                                                     cudaGraphicsRegisterFlags flags = cudaGraphicsRegisterFlagsNone) {
        auto buffer = std::make_shared<CudaGLBuffer<T>>(bufferId, count, flags);
        return buffer;
    }
    
    // Check if CUDA-OpenGL interop is supported on this device
    bool isGLInteropSupported();
    
    // Synchronize device
    void synchronize() {
        CUDA_CHECK(cudaDeviceSynchronize());
    }
    
    // Get device properties
    cudaDeviceProp getDeviceProperties();
};

} // namespace drumforge

#endif // DRUMFORGE_CUDA_MEMORY_MANAGER_H