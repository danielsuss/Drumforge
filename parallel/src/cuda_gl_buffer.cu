#include "cuda_gl_buffer.h"
#include "cuda_memory_manager.h"
#include <iostream>

namespace drumforge {

// Constructor
CudaGLBuffer::CudaGLBuffer()
    : glBufferId(0), cudaResource(nullptr), devicePtr(nullptr),
      elementCount(0), byteSize(0), elementSize(0), isMapped(false) {
}

// Destructor
CudaGLBuffer::~CudaGLBuffer() {
    unregisterBuffer();
}

// Move constructor
CudaGLBuffer::CudaGLBuffer(CudaGLBuffer&& other) noexcept 
    : glBufferId(other.glBufferId), cudaResource(other.cudaResource),
      devicePtr(other.devicePtr), elementCount(other.elementCount),
      byteSize(other.byteSize), elementSize(other.elementSize), isMapped(other.isMapped) {
    other.glBufferId = 0;
    other.cudaResource = nullptr;
    other.devicePtr = nullptr;
    other.elementCount = 0;
    other.byteSize = 0;
    other.elementSize = 0;
    other.isMapped = false;
}

// Move assignment operator
CudaGLBuffer& CudaGLBuffer::operator=(CudaGLBuffer&& other) noexcept {
    if (this != &other) {
        unregisterBuffer();
        glBufferId = other.glBufferId;
        cudaResource = other.cudaResource;
        devicePtr = other.devicePtr;
        elementCount = other.elementCount;
        byteSize = other.byteSize;
        elementSize = other.elementSize;
        isMapped = other.isMapped;
        other.glBufferId = 0;
        other.cudaResource = nullptr;
        other.devicePtr = nullptr;
        other.elementCount = 0;
        other.byteSize = 0;
        other.elementSize = 0;
        other.isMapped = false;
    }
    return *this;
}

// Register an OpenGL buffer with CUDA
void CudaGLBuffer::registerBuffer(GLuint bufferId, size_t count, size_t elemSize, 
                                cudaGraphicsRegisterFlags flags) {
    // Unregister any existing buffer first
    unregisterBuffer();
    
    glBufferId = bufferId;
    elementCount = count;
    elementSize = elemSize;
    byteSize = count * elemSize;
    
    // Register the buffer with CUDA
    cudaError_t error = cudaGraphicsGLRegisterBuffer(&cudaResource, glBufferId, flags);
    if (error != cudaSuccess) {
        throw CudaException("Failed to register OpenGL buffer with CUDA: " + 
                           std::string(cudaGetErrorString(error)));
    }
}

// Unregister the buffer
void CudaGLBuffer::unregisterBuffer() {
    if (isMapped) {
        unmap();
    }
    
    if (cudaResource != nullptr) {
        cudaError_t error = cudaGraphicsUnregisterResource(cudaResource);
        if (error != cudaSuccess) {
            std::cerr << "Warning: Failed to unregister CUDA resource: " 
                      << cudaGetErrorString(error) << std::endl;
        }
        cudaResource = nullptr;
    }
    
    glBufferId = 0;
    elementCount = 0;
    byteSize = 0;
    elementSize = 0;
}

// Map the buffer for CUDA access
void* CudaGLBuffer::map() {
    if (!cudaResource) {
        throw CudaException("Cannot map unregistered buffer");
    }
    
    if (!isMapped) {
        // Synchronize before mapping to ensure any previous GPU operations are complete
        cudaError_t error = cudaDeviceSynchronize();
        if (error != cudaSuccess) {
            throw CudaException("Failed to synchronize device before mapping: " + 
                               std::string(cudaGetErrorString(error)));
        }
        
        error = cudaGraphicsMapResources(1, &cudaResource);
        if (error != cudaSuccess) {
            throw CudaException("Failed to map CUDA resource: " + 
                               std::string(cudaGetErrorString(error)));
        }
        
        size_t mappedSize;
        error = cudaGraphicsResourceGetMappedPointer(&devicePtr, &mappedSize, cudaResource);
        if (error != cudaSuccess) {
            // Unmap the resource to avoid leaking
            cudaGraphicsUnmapResources(1, &cudaResource);
            // Make sure isMapped stays false
            isMapped = false;
            throw CudaException("Failed to get mapped pointer: " + 
                               std::string(cudaGetErrorString(error)));
        }
        
        isMapped = true;
        
        // Check if the mapped size matches what we expect
        if (mappedSize != byteSize) {
            std::cerr << "Warning: Mapped size (" << mappedSize << ") doesn't match expected size (" 
                      << byteSize << ")" << std::endl;
        }
    }
    
    return devicePtr;
}

// Unmap the buffer
void CudaGLBuffer::unmap() {
    if (cudaResource && isMapped) {
        // Synchronize before unmapping to ensure all CUDA operations are complete
        cudaError_t error = cudaDeviceSynchronize();
        if (error != cudaSuccess) {
            throw CudaException("Failed to synchronize device before unmapping: " + 
                               std::string(cudaGetErrorString(error)));
        }
        
        error = cudaGraphicsUnmapResources(1, &cudaResource);
        if (error != cudaSuccess) {
            throw CudaException("Failed to unmap CUDA resource: " + 
                               std::string(cudaGetErrorString(error)));
        }
        isMapped = false;
        devicePtr = nullptr;
    }
}

// Get device pointer
void* CudaGLBuffer::getDevicePtr() const {
    if (!isMapped) {
        throw CudaException("Buffer not mapped, cannot access device pointer");
    }
    return devicePtr;
}

namespace cuda_gl {

// Check if CUDA-GL interop is supported
bool isInteropSupported() {
    int deviceCount = 0;
    cudaError_t error = cudaGetDeviceCount(&deviceCount);
    if (error != cudaSuccess || deviceCount == 0) {
        return false;
    }
    
    int currentDevice = 0;
    error = cudaGetDevice(&currentDevice);
    if (error != cudaSuccess) {
        return false;
    }
    
    cudaDeviceProp props;
    error = cudaGetDeviceProperties(&props, currentDevice);
    if (error != cudaSuccess) {
        return false;
    }
    
    // OpenGL interop requires compute capability 1.1 or higher and canMapHostMemory
    return (props.major > 1 || (props.major == 1 && props.minor >= 1)) && props.canMapHostMemory;
}

// Create an OpenGL buffer and register it with CUDA
std::shared_ptr<CudaGLBuffer> createBuffer(size_t count, size_t elemSize, 
                                          GLenum target, GLenum usage) {
    // Verify we have a valid OpenGL context
    if (glGetError() != GL_NO_ERROR) {
        throw CudaException("OpenGL error detected before creating buffer");
    }
    
    // Create an OpenGL buffer
    GLuint bufferId = 0;
    glGenBuffers(1, &bufferId);
    glBindBuffer(target, bufferId);
    glBufferData(target, count * elemSize, nullptr, usage);
    glBindBuffer(target, 0);
    
    // Check for errors in buffer creation
    GLenum glError = glGetError();
    if (glError != GL_NO_ERROR) {
        glDeleteBuffers(1, &bufferId);
        throw CudaException("Failed to create OpenGL buffer, error code: " + 
                           std::to_string(glError));
    }
    
    // Register it with CUDA
    try {
        auto buffer = std::make_shared<CudaGLBuffer>();
        buffer->registerBuffer(bufferId, count, elemSize);
        return buffer;
    } catch (const CudaException& e) {
        // Clean up the OpenGL buffer if CUDA registration fails
        glDeleteBuffers(1, &bufferId);
        throw;
    }
}

// Register an existing OpenGL buffer with CUDA
std::shared_ptr<CudaGLBuffer> registerBuffer(GLuint bufferId, size_t count, size_t elemSize, 
                                           cudaGraphicsRegisterFlags flags) {
    auto buffer = std::make_shared<CudaGLBuffer>();
    buffer->registerBuffer(bufferId, count, elemSize, flags);
    return buffer;
}

} // namespace cuda_gl

} // namespace drumforge