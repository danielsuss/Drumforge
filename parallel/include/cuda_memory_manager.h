#ifndef DRUMFORGE_CUDA_MEMORY_MANAGER_H
#define DRUMFORGE_CUDA_MEMORY_MANAGER_H

#include <cuda_runtime.h>
#include <vector>
#include <string>
#include <memory>
#include <stdexcept>

// Forward declarations for CUDA-OpenGL interop
typedef unsigned int GLuint;

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
    bool mapped;
    
public:
    CudaBuffer() : devicePtr(nullptr), elementCount(0), byteSize(0), mapped(false) {}
    
    CudaBuffer(size_t count) : devicePtr(nullptr), elementCount(count), 
                            byteSize(count * sizeof(T)), mapped(false) {
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
          byteSize(other.byteSize), mapped(other.mapped) {
        other.devicePtr = nullptr;
        other.elementCount = 0;
        other.byteSize = 0;
        other.mapped = false;
    }
    
    CudaBuffer& operator=(CudaBuffer&& other) noexcept {
        if (this != &other) {
            release();
            devicePtr = other.devicePtr;
            elementCount = other.elementCount;
            byteSize = other.byteSize;
            mapped = other.mapped;
            other.devicePtr = nullptr;
            other.elementCount = 0;
            other.byteSize = 0;
            other.mapped = false;
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
        mapped = false;
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
    bool isMapped() const { return mapped; }
    
    // Set mapped status (used by memory manager)
    void setMapped(bool status) { mapped = status; }
};

// Class for CUDA-OpenGL interoperability buffer
class CudaGLBuffer {
private:
    struct cudaGraphicsResource* resource;
    size_t size;
    GLuint glBuffer;
    bool mapped;
    
public:
    CudaGLBuffer() : resource(nullptr), size(0), glBuffer(0), mapped(false) {}
    
    ~CudaGLBuffer() {
        unregister();
    }
    
    // Register an OpenGL buffer with CUDA
    void registerBuffer(GLuint buffer, unsigned int flags) {
        unregister();
        glBuffer = buffer;
        CUDA_CHECK(cudaGraphicsGLRegisterBuffer(&resource, buffer, flags));
    }
    
    // Unregister the buffer
    void unregister() {
        if (resource != nullptr) {
            if (mapped) {
                CUDA_CHECK(cudaGraphicsUnmapResources(1, &resource));
                mapped = false;
            }
            CUDA_CHECK(cudaGraphicsUnregisterResource(resource));
            resource = nullptr;
        }
        glBuffer = 0;
    }
    
    // Map for access from CUDA
    void* map() {
        if (resource == nullptr) {
            throw CudaException("Cannot map unregistered GL buffer");
        }
        
        void* devicePtr = nullptr;
        
        CUDA_CHECK(cudaGraphicsMapResources(1, &resource));
        CUDA_CHECK(cudaGraphicsResourceGetMappedPointer(&devicePtr, &size, resource));
        
        mapped = true;
        return devicePtr;
    }
    
    // Unmap after CUDA access
    void unmap() {
        if (resource == nullptr || !mapped) {
            return;
        }
        
        CUDA_CHECK(cudaGraphicsUnmapResources(1, &resource));
        mapped = false;
    }
    
    // Getters
    GLuint getGLBuffer() const { return glBuffer; }
    size_t getSize() const { return size; }
    bool isMapped() const { return mapped; }
};

// Main memory manager class
class CudaMemoryManager {
private:
    // Track all allocated buffers for potential cleanup
    std::vector<void*> allocatedBuffers;
    // Track all registered GL buffers
    std::vector<CudaGLBuffer*> registeredGLBuffers;
    
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
    
    // Register an OpenGL buffer for CUDA interop
    std::shared_ptr<CudaGLBuffer> registerGLBuffer(GLuint buffer, unsigned int flags);
    
    // Synchronize device
    void synchronize() {
        CUDA_CHECK(cudaDeviceSynchronize());
    }
    
    // Get device properties
    cudaDeviceProp getDeviceProperties();
};

} // namespace drumforge

#endif // DRUMFORGE_CUDA_MEMORY_MANAGER_H