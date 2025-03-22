#ifndef DRUMFORGE_CUDA_GL_BUFFER_H
#define DRUMFORGE_CUDA_GL_BUFFER_H

// Include OpenGL headers first
#include <GL/glew.h>

// Then include CUDA headers
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

#include <string>
#include <memory>

namespace drumforge {

// Forward declaration of the exception class
class CudaException;

/**
 * @brief Class for managing a CUDA-OpenGL interoperable buffer
 * 
 * This class handles the registration, mapping, and unmapping of
 * OpenGL buffers for use with CUDA.
 */
class CudaGLBuffer {
private:
    GLuint glBufferId;                    // OpenGL buffer ID
    cudaGraphicsResource_t cudaResource;  // CUDA graphics resource
    void* devicePtr;                      // Device pointer when mapped
    size_t elementCount;                  // Number of elements
    size_t byteSize;                      // Size in bytes
    size_t elementSize;                   // Size of each element
    bool isMapped;                        // Whether buffer is currently mapped

public:
    // Constructor and destructor
    CudaGLBuffer();
    ~CudaGLBuffer();
    
    // Prevent copying
    CudaGLBuffer(const CudaGLBuffer&) = delete;
    CudaGLBuffer& operator=(const CudaGLBuffer&) = delete;
    
    // Allow moving
    CudaGLBuffer(CudaGLBuffer&& other) noexcept;
    CudaGLBuffer& operator=(CudaGLBuffer&& other) noexcept;
    
    // Register an OpenGL buffer with CUDA
    void registerBuffer(GLuint bufferId, size_t count, size_t elemSize, 
                        cudaGraphicsRegisterFlags flags = cudaGraphicsRegisterFlagsNone);
    
    // Unregister the buffer
    void unregisterBuffer();
    
    // Map the buffer for CUDA access
    void* map();
    
    // Unmap the buffer
    void unmap();
    
    // Getters
    void* getDevicePtr() const;
    GLuint getGLBufferId() const { return glBufferId; }
    size_t size() const { return elementCount; }
    size_t bytes() const { return byteSize; }
    bool isRegistered() const { return cudaResource != nullptr; }
    bool isMappedForCUDA() const { return isMapped; }
};

/**
 * @brief Utility functions for CUDA-OpenGL interoperability
 */
namespace cuda_gl {
    // Check if CUDA-GL interop is supported
    bool isInteropSupported();
    
    // Create an OpenGL buffer and register it with CUDA
    std::shared_ptr<CudaGLBuffer> createBuffer(size_t count, size_t elemSize, 
                                              GLenum target = GL_ARRAY_BUFFER, 
                                              GLenum usage = GL_DYNAMIC_DRAW);
    
    // Register an existing OpenGL buffer with CUDA
    std::shared_ptr<CudaGLBuffer> registerBuffer(GLuint bufferId, size_t count, size_t elemSize, 
                                                cudaGraphicsRegisterFlags flags = cudaGraphicsRegisterFlagsNone);
}

} // namespace drumforge

#endif // DRUMFORGE_CUDA_GL_BUFFER_H