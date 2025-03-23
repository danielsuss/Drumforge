#include "membrane_kernels.cuh"
#include "cuda_memory_manager.h"
#include <cmath>
#include <iostream>

namespace drumforge {

//-----------------------------------------------------------------------------
// CUDA Kernel Implementations
//-----------------------------------------------------------------------------

__global__ void initializeCircleMaskKernel(int* mask, const MembraneKernelParams params) {
    // Calculate thread's position in the membrane grid
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    // Check bounds
    if (x >= params.membraneWidth || y >= params.membraneHeight) {
        return;
    }
    
    // Calculate distance from center
    float dx = x - params.centerX;
    float dy = y - params.centerY;
    float distSquared = dx * dx + dy * dy;
    
    // Set mask value (1 if inside circle, 0 if outside)
    mask[getIndex(x, y, params.membraneWidth)] = (distSquared <= params.radiusSquared) ? 1 : 0;
}

__global__ void resetMembraneKernel(float* heights, float* prevHeights, float* velocities, 
                                  const int* mask, const MembraneKernelParams params) {
    // Calculate thread's position in the membrane grid
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    // Check bounds
    if (x >= params.membraneWidth || y >= params.membraneHeight) {
        return;
    }
    
    // Get linear index into arrays
    int idx = getIndex(x, y, params.membraneWidth);
    
    // Only process points within the circular membrane
    if (mask[idx]) {
        heights[idx] = 0.0f;
        prevHeights[idx] = 0.0f;
        velocities[idx] = 0.0f;
    }
}

__global__ void updateMembraneKernel(float* heights, float* prevHeights, float* velocities,
                                   const int* mask, const MembraneKernelParams params,
                                   float timestep) {
    // Calculate thread's position in the membrane grid
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    // Check bounds and ensure we're at least one cell away from the boundary
    // This is because we need to access neighboring cells
    if (x < 1 || x >= params.membraneWidth - 1 || y < 1 || y >= params.membraneHeight - 1) {
        return;
    }
    
    // Get linear index into arrays
    int idx = getIndex(x, y, params.membraneWidth);
    
    // Only process points within the circular membrane
    if (mask[idx]) {
        // Get the current height
        float current = heights[idx];
        float prev = prevHeights[idx];
        
        // Get neighboring heights
        float left = heights[getIndex(x-1, y, params.membraneWidth)];
        float right = heights[getIndex(x+1, y, params.membraneWidth)];
        float up = heights[getIndex(x, y-1, params.membraneWidth)];
        float down = heights[getIndex(x, y+1, params.membraneWidth)];
        
        // Calculate the Laplacian (discretized ∇²u term)
        float laplacian = left + right + up + down - 4.0f * current;
        
        // Space discretization (normalized to 1.0)
        float dx = 1.0f;
        
        // Calculate the coefficient for the wave equation
        float c = params.waveSpeed;
        float coef = c * c * timestep * timestep / (dx * dx);
        
        // Calculate the new height using FDTD update rule
        // Special case for first timestep (prevTimestep would be 0 in component)
        float new_height;
        
        if (prev == current) {  // Indicates first timestep
            // For first timestep: u(t+dt) = u(t) + dt*v(t) + 0.5*dt²*a(t)
            // where a(t) = c²*∇²u - damping*v(t)
            float velocity = velocities[idx];
            float acceleration = c * c * laplacian / (dx * dx) - params.damping * velocity;
            new_height = current + timestep * velocity + 0.5f * timestep * timestep * acceleration;
        } else {
            // Standard leapfrog update for wave equation:
            // u(t+dt) = 2*u(t) - u(t-dt) + c²*(dt/dx)²*∇²u(t)
            new_height = 2.0f * current - prev + coef * laplacian;
            
            // Apply damping term
            // Use a damping term proportional to first time derivative of u
            float velocity_approx = (current - prev) / timestep;
            new_height -= params.damping * velocity_approx * timestep;
        }
        
        // Update the velocity using central difference approximation
        velocities[idx] = (new_height - prev) / (2.0f * timestep);
        
        // Store the new height in the output array
        // Note: We're using the current height as the previous height for the next step
        prevHeights[idx] = current;
        heights[idx] = new_height;
    }
}

__global__ void applyImpulseKernel(float* heights, const int* mask, 
                                 const MembraneKernelParams params,
                                 float x, float y, float strength) {
    // Calculate thread's position in the membrane grid
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;
    
    // Check bounds
    if (ix >= params.membraneWidth || iy >= params.membraneHeight) {
        return;
    }
    
    // Convert normalized position [0,1] to grid coordinates
    int centerX = static_cast<int>(x * params.membraneWidth);
    int centerY = static_cast<int>(y * params.membraneHeight);
    
    // Apply impulse only if this point is within the membrane
    int idx = getIndex(ix, iy, params.membraneWidth);
    if (mask[idx]) {
        // Calculate distance from the impulse center
        float dx = ix - centerX;
        float dy = iy - centerY;
        float distSquared = dx * dx + dy * dy;
        
        // Size of the impulse (radius squared in grid cells)
        const float impulseRadiusSquared = 25.0f;  // 5² = 25
        
        // Only apply impulse within the impulse radius
        if (distSquared <= impulseRadiusSquared) {
            // Apply a Gaussian-shaped impulse
            float dist = sqrtf(distSquared);
            float falloff = expf(-dist*dist / (impulseRadiusSquared/4.0f));
            
            // Apply the impulse - negative for downward strike
            heights[idx] -= strength * falloff;
        }
    }
}

__global__ void updateVisualizationVerticesKernel(float3* vertices, const float* heights,
                                               const int* mask, const MembraneKernelParams params,
                                               float scale) {
    // Calculate thread's position in the membrane grid
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    // Check bounds
    if (x >= params.membraneWidth || y >= params.membraneHeight) {
        return;
    }
    
    // Get linear index into arrays
    int idx = getIndex(x, y, params.membraneWidth);
    
    // Create vertex at this grid position
    // Scale x/y to be centered at 0,0
    float xPos = x - params.centerX;
    float yPos = y - params.centerY;
    
    // Get height and apply scale factor for visualization
    float zPos = 0.0f;
    
    // Only show height for points inside membrane
    if (mask[idx]) {
        zPos = heights[idx] * scale;
    }
    
    // Update vertex position
    vertices[idx] = make_float3(xPos, yPos, zPos);
}

//-----------------------------------------------------------------------------
// Host-side wrapper functions
//-----------------------------------------------------------------------------

void initializeCircleMask(int* d_mask, const MembraneKernelParams& params) {
    // Calculate optimal block size for the kernel
    dim3 blockSize = calculateOptimalBlockSize(params.membraneWidth, params.membraneHeight);
    
    // Calculate grid size based on membrane dimensions and block size
    dim3 gridSize(
        (params.membraneWidth + blockSize.x - 1) / blockSize.x,
        (params.membraneHeight + blockSize.y - 1) / blockSize.y
    );
    
    // Launch kernel
    initializeCircleMaskKernel<<<gridSize, blockSize>>>(d_mask, params);
    
    // Check for errors
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        throw CudaException("Failed to initialize circle mask: " + 
                           std::string(cudaGetErrorString(error)));
    }
}

void resetMembrane(float* d_heights, float* d_prevHeights, float* d_velocities, 
                 const int* d_mask, const MembraneKernelParams& params) {
    // Calculate optimal block size for the kernel
    dim3 blockSize = calculateOptimalBlockSize(params.membraneWidth, params.membraneHeight);
    
    // Calculate grid size based on membrane dimensions and block size
    dim3 gridSize(
        (params.membraneWidth + blockSize.x - 1) / blockSize.x,
        (params.membraneHeight + blockSize.y - 1) / blockSize.y
    );
    
    // Launch kernel
    resetMembraneKernel<<<gridSize, blockSize>>>(
        d_heights, d_prevHeights, d_velocities, d_mask, params
    );
    
    // Check for errors
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        throw CudaException("Failed to reset membrane: " + 
                           std::string(cudaGetErrorString(error)));
    }
}

void updateMembrane(float* d_heights, float* d_prevHeights, float* d_velocities,
                  const int* d_mask, const MembraneKernelParams& params, float timestep) {
    // Calculate optimal block size for the kernel
    dim3 blockSize = calculateOptimalBlockSize(params.membraneWidth, params.membraneHeight);
    
    // Calculate grid size based on membrane dimensions and block size
    dim3 gridSize(
        (params.membraneWidth + blockSize.x - 1) / blockSize.x,
        (params.membraneHeight + blockSize.y - 1) / blockSize.y
    );
    
    // Launch kernel
    updateMembraneKernel<<<gridSize, blockSize>>>(
        d_heights, d_prevHeights, d_velocities, d_mask, params, timestep
    );
    
    // Check for errors
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        throw CudaException("Failed to update membrane: " + 
                           std::string(cudaGetErrorString(error)));
    }
}

void applyImpulse(float* d_heights, const int* d_mask, const MembraneKernelParams& params,
               float x, float y, float strength) {
    // Calculate optimal block size for the kernel
    dim3 blockSize = calculateOptimalBlockSize(params.membraneWidth, params.membraneHeight);
    
    // Calculate grid size based on membrane dimensions and block size
    dim3 gridSize(
        (params.membraneWidth + blockSize.x - 1) / blockSize.x,
        (params.membraneHeight + blockSize.y - 1) / blockSize.y
    );
    
    // Launch kernel
    applyImpulseKernel<<<gridSize, blockSize>>>(
        d_heights, d_mask, params, x, y, strength
    );
    
    // Check for errors
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        throw CudaException("Failed to apply impulse: " + 
                           std::string(cudaGetErrorString(error)));
    }
}

void updateVisualizationVertices(float3* d_vertices, const float* d_heights,
                              const int* d_mask, const MembraneKernelParams& params, float scale) {
    // Calculate optimal block size for the kernel
    dim3 blockSize = calculateOptimalBlockSize(params.membraneWidth, params.membraneHeight);
    
    // Calculate grid size based on membrane dimensions and block size
    dim3 gridSize(
        (params.membraneWidth + blockSize.x - 1) / blockSize.x,
        (params.membraneHeight + blockSize.y - 1) / blockSize.y
    );
    
    // Launch kernel
    updateVisualizationVerticesKernel<<<gridSize, blockSize>>>(
        d_vertices, d_heights, d_mask, params, scale
    );
    
    // Check for errors
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        throw CudaException("Failed to update visualization vertices: " + 
                           std::string(cudaGetErrorString(error)));
    }
}

float calculateStableTimestep(const MembraneKernelParams& params) {
    // Calculate wave propagation speed (c)
    float c = params.waveSpeed;
    
    // Grid spacing (normalized to 1.0 in our simulation)
    float dx = 1.0f;
    
    // CFL condition for 2D wave equation: dt <= dx / (c * sqrt(2))
    float cflTimestep = dx / (c * sqrtf(2.0f));
    
    // Use 80% of CFL condition for safety
    return 0.8f * cflTimestep;
}

dim3 calculateOptimalBlockSize(int dataWidth, int dataHeight) {
    // Default block size for 2D grid (8x8 = 64 threads per block)
    // This is a conservative choice that works well on most GPUs
    int blockSizeX = 8;
    int blockSizeY = 8;
    
    // For very small membranes, use smaller blocks
    if (dataWidth < 16 || dataHeight < 16) {
        blockSizeX = 4;
        blockSizeY = 4;
    }
    // For larger membranes, use larger blocks to increase parallelism
    else if (dataWidth >= 128 && dataHeight >= 128) {
        blockSizeX = 16;
        blockSizeY = 16;
    }
    
    return dim3(blockSizeX, blockSizeY);
}

} // namespace drumforge