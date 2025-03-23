#ifndef DRUMFORGE_MEMBRANE_KERNELS_CUH
#define DRUMFORGE_MEMBRANE_KERNELS_CUH

#include <cuda_runtime.h>
#include <vector_types.h>
#include <glm/glm.hpp>

namespace drumforge {

/**
 * Structure containing parameters for membrane simulation kernels
 * Used to avoid passing multiple arguments to each kernel
 */
struct MembraneKernelParams {
    // Membrane data dimensions
    int membraneWidth;   // Width of the membrane data array
    int membraneHeight;  // Height of the membrane data array
    
    // Membrane physical properties
    float radius;    // Radius of the membrane in grid units
    float tension;   // Membrane tension
    float damping;   // Damping coefficient
    
    // Center point of the membrane in membrane coordinates
    float centerX;
    float centerY;
    
    // Derived parameters
    float radiusSquared;  // Pre-computed radius squared for faster circle tests
    float waveSpeed;      // Pre-computed wave speed derived from tension
    
    // Constructor to initialize with defaults
    MembraneKernelParams() 
        : membraneWidth(0), membraneHeight(0), radius(0.0f), tension(1.0f), damping(0.01f),
          centerX(0.0f), centerY(0.0f), radiusSquared(0.0f), waveSpeed(0.0f) {}
};

/**
 * Initialize the circular mask for the membrane
 * 
 * @param mask Output mask (1 for points inside circle, 0 outside)
 * @param params Kernel parameters
 */
__global__ void initializeCircleMaskKernel(int* mask, const MembraneKernelParams params);

/**
 * Reset the membrane to its flat state
 * 
 * @param heights Current heights buffer
 * @param prevHeights Previous heights buffer
 * @param velocities Velocities buffer
 * @param mask Circular mask
 * @param params Kernel parameters
 */
__global__ void resetMembraneKernel(float* heights, float* prevHeights, float* velocities, 
                                   const int* mask, const MembraneKernelParams params);

/**
 * Update the membrane heights using FDTD wave equation
 * 
 * @param heights Current heights buffer
 * @param prevHeights Previous heights buffer (will be updated to current heights)
 * @param velocities Velocities buffer (will be updated)
 * @param mask Circular mask
 * @param params Kernel parameters
 * @param timestep Simulation time step
 */
__global__ void updateMembraneKernel(float* heights, float* prevHeights, float* velocities,
                                    const int* mask, const MembraneKernelParams params,
                                    float timestep);

/**
 * Apply a drum strike impulse at a specific position
 * 
 * @param heights Heights buffer to modify
 * @param mask Circular mask
 * @param params Kernel parameters
 * @param x Normalized x position [0,1]
 * @param y Normalized y position [0,1]
 * @param strength Impulse strength
 */
__global__ void applyImpulseKernel(float* heights, const int* mask, 
                                  const MembraneKernelParams params,
                                  float x, float y, float strength);

/**
 * Update the visualization vertices for rendering
 * 
 * @param vertices Output vertices buffer (interop with OpenGL)
 * @param heights Current heights
 * @param mask Circular mask
 * @param params Kernel parameters
 * @param scale Height scale factor for visualization
 */
__global__ void updateVisualizationVerticesKernel(float3* vertices, const float* heights,
                                                const int* mask, const MembraneKernelParams params,
                                                float scale);

/**
 * Calculate a 1D index from 2D coordinates
 * 
 * @param x X coordinate
 * @param y Y coordinate
 * @param width Width of the data array
 * @return 1D index
 */
__device__ __forceinline__ int getIndex(int x, int y, int width) {
    return y * width + x;
}

/**
 * Check if a point is inside the circular membrane
 * 
 * @param x X coordinate
 * @param y Y coordinate
 * @param params Kernel parameters
 * @return True if inside, false if outside
 */
__device__ __forceinline__ bool isInsideCircle(int x, int y, const MembraneKernelParams& params) {
    float dx = x - params.centerX;
    float dy = y - params.centerY;
    float distSquared = dx * dx + dy * dy;
    return distSquared <= params.radiusSquared;
}

// Host-side function to launch the initialization kernel
void initializeCircleMask(int* d_mask, const MembraneKernelParams& params);

// Host-side function to launch the reset kernel
void resetMembrane(float* d_heights, float* d_prevHeights, float* d_velocities, 
                  const int* d_mask, const MembraneKernelParams& params);

// Host-side function to launch the update kernel
void updateMembrane(float* d_heights, float* d_prevHeights, float* d_velocities,
                   const int* d_mask, const MembraneKernelParams& params, float timestep);

// Host-side function to launch the impulse kernel
void applyImpulse(float* d_heights, const int* d_mask, const MembraneKernelParams& params,
                 float x, float y, float strength);

// Host-side function to launch the visualization update kernel
void updateVisualizationVertices(float3* d_vertices, const float* d_heights,
                               const int* d_mask, const MembraneKernelParams& params, float scale);

// Helper to compute a stable time step for the membrane simulation
float calculateStableTimestep(const MembraneKernelParams& params);

// Helper to calculate optimal CUDA thread and block configuration 
dim3 calculateOptimalBlockSize(int dataWidth, int dataHeight);

} // namespace drumforge

#endif // DRUMFORGE_MEMBRANE_KERNELS_CUH