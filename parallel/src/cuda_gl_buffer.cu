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
    
    // Validate input
    if (bufferId == 0) {
        throw CudaException("Invalid OpenGL buffer ID (0) provided for CUDA registration");
    }
    
    glBufferId = bufferId;
    elementCount = count;
    elementSize = elemSize;
    byteSize = count * elemSize;
    
    // Ensure the buffer exists in OpenGL
    GLint prevBuffer;
    glGetIntegerv(GL_ARRAY_BUFFER_BINDING, &prevBuffer);
    
    glBindBuffer(GL_ARRAY_BUFFER, glBufferId);
    if (glGetError() != GL_NO_ERROR) {
        throw CudaException("OpenGL buffer does not exist or cannot be bound");
    }
    
    // Check buffer size in OpenGL
    GLint bufferSize;
    glGetBufferParameteriv(GL_ARRAY_BUFFER, GL_BUFFER_SIZE, &bufferSize);
    if (bufferSize < static_cast<GLint>(byteSize)) {
        glBindBuffer(GL_ARRAY_BUFFER, prevBuffer);
        throw CudaException("OpenGL buffer size is smaller than requested size for CUDA");
    }
    
    // Restore previous buffer binding
    glBindBuffer(GL_ARRAY_BUFFER, prevBuffer);
    
    // Register the buffer with CUDA
    std::cout << "Registering OpenGL buffer " << glBufferId << " with CUDA" << std::endl;
    cudaError_t error = cudaGraphicsGLRegisterBuffer(&cudaResource, glBufferId, flags);
    
    if (error != cudaSuccess) {
        throw CudaException("Failed to register OpenGL buffer with CUDA: " + 
                           std::string(cudaGetErrorString(error)));
    }
    
    std::cout << "Successfully registered OpenGL buffer with CUDA" << std::endl;
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
        std::cout << "Mapping OpenGL buffer " << glBufferId << " for CUDA access..." << std::endl;
        
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
        
        std::cout << "Buffer mapped successfully, device pointer: " << devicePtr << std::endl;
    }
    
    return devicePtr;
}

// Unmap the buffer
void CudaGLBuffer::unmap() {
    if (cudaResource && isMapped) {
        std::cout << "Unmapping OpenGL buffer " << glBufferId << " from CUDA..." << std::endl;
        
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
        
        std::cout << "Buffer unmapped successfully" << std::endl;
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
    
    // Print more detailed interop capability information
    std::cout << "CUDA Interop Capability Check:" << std::endl;
    std::cout << "  Compute Capability: " << props.major << "." << props.minor << std::endl;
    std::cout << "  canMapHostMemory: " << (props.canMapHostMemory ? "Yes" : "No") << std::endl;
    
    // Additional checks that might be relevant for interop
    // Note: Driver version and OpenGL version can also affect interop support,
    // but these are harder to check programmatically
    
    // OpenGL interop requires compute capability 1.1 or higher and typically canMapHostMemory
    return (props.major > 1 || (props.major == 1 && props.minor >= 1)) && props.canMapHostMemory;
}

// Create an OpenGL buffer and register it with CUDA
std::shared_ptr<CudaGLBuffer> createBuffer(size_t count, size_t elemSize, 
                                          GLenum target, GLenum usage) {
    // Verify we have a valid OpenGL context
    GLenum glError = glGetError();
    if (glError != GL_NO_ERROR) {
        throw CudaException("OpenGL error detected before creating buffer: " + 
                           std::to_string(glError));
    }
    
    // Create an OpenGL buffer
    GLuint bufferId = 0;
    glGenBuffers(1, &bufferId);
    if (bufferId == 0) {
        throw CudaException("Failed to generate OpenGL buffer");
    }
    
    glBindBuffer(target, bufferId);
    
    // Print buffer creation info
    std::cout << "Creating OpenGL buffer " << bufferId
              << " with " << count << " elements of size " << elemSize
              << " (total " << (count * elemSize) << " bytes)" << std::endl;
    
    glBufferData(target, count * elemSize, nullptr, usage);
    
    // Check for errors in buffer creation
    glError = glGetError();
    if (glError != GL_NO_ERROR) {
        glDeleteBuffers(1, &bufferId);
        throw CudaException("Failed to create OpenGL buffer, error code: " + 
                           std::to_string(glError));
    }
    
    // Additional validation: check that the buffer was created with the correct size
    GLint bufferSize;
    glGetBufferParameteriv(target, GL_BUFFER_SIZE, &bufferSize);
    if (bufferSize != static_cast<GLint>(count * elemSize)) {
        glDeleteBuffers(1, &bufferId);
        throw CudaException("OpenGL buffer created with incorrect size: " +
                           std::to_string(bufferSize) + " (expected " +
                           std::to_string(count * elemSize) + ")");
    }
    
    // Unbind the buffer
    glBindBuffer(target, 0);
    
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