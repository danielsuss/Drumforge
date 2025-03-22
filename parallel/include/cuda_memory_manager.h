#ifndef DRUMFORGE_CUDA_MEMORY_MANAGER_H
#define DRUMFORGE_CUDA_MEMORY_MANAGER_H

#include <cuda_runtime.h>
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
    
    // Synchronize device
    void synchronize() {
        CUDA_CHECK(cudaDeviceSynchronize());
    }
    
    // Get device properties
    cudaDeviceProp getDeviceProperties();
};

} // namespace drumforge

#endif // DRUMFORGE_CUDA_MEMORY_MANAGER_H