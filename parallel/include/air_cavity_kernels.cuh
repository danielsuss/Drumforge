#ifndef DRUMFORGE_AIR_CAVITY_KERNELS_CUH
#define DRUMFORGE_AIR_CAVITY_KERNELS_CUH

#include <cuda_runtime.h>
#include <vector_types.h>
#include <glm/glm.hpp>

namespace drumforge {

/**
 * Structure containing parameters for air cavity simulation kernels
 * Used to avoid passing multiple arguments to each kernel
 */
struct AirCavityKernelParams {
    // Grid dimensions
    int cavityWidth;    // Width of the cavity data array (X dimension)
    int cavityHeight;   // Height of the cavity data array (Y dimension)
    int cavityDepth;    // Depth of the cavity data array (Z dimension)
    
    // Physical properties
    float speedOfSound;       // Speed of sound in air (m/s)
    float density;            // Density of air (kg/m^3)
    float dampingCoefficient; // Damping coefficient for stability and absorption
    
    // Grid cell size in physical units
    float cellSize;           // Size of each grid cell (m)
    
    // Derived parameters
    float courantNumber;      // Courant number for stability (should be < 1/sqrt(3) for 3D)
    
    // Constructor to initialize with defaults
    AirCavityKernelParams() 
        : cavityWidth(0), cavityHeight(0), cavityDepth(0),
          speedOfSound(343.0f), density(1.2f), dampingCoefficient(0.001f),
          cellSize(1.0f), courantNumber(0.5f) {}
};

/**
 * Kernel to initialize the pressure and velocity fields to zero
 */
__global__ void initializeFieldsKernel(float* pressure, float* prevPressure, 
                                    float* velocityX, float* velocityY, float* velocityZ,
                                    const AirCavityKernelParams params);

/**
 * Kernel to update velocity fields based on pressure gradients
 * Using a staggered grid approach for better stability
 */
__global__ void updateVelocityKernel(float* pressure, 
                                    float* velocityX, float* velocityY, float* velocityZ,
                                    const AirCavityKernelParams params, float timestep);

/**
 * Kernel to update pressure field based on velocity divergence
 */
__global__ void updatePressureKernel(float* pressure, float* prevPressure,
                                    float* velocityX, float* velocityY, float* velocityZ,
                                    const AirCavityKernelParams params, float timestep);

/**
 * Kernel to apply boundary conditions to the simulation
 * This handles reflective boundaries at the edges of the simulation
 */
__global__ void applyBoundaryConditionsKernel(float* pressure,
                                            float* velocityX, float* velocityY, float* velocityZ,
                                            const AirCavityKernelParams params);

/**
 * Kernel to add a pressure impulse at a specific position
 */
__global__ void addPressureImpulseKernel(float* pressure, 
                                        const AirCavityKernelParams params,
                                        float x, float y, float z, 
                                        float strength, float radius);

/**
 * Kernel to sample pressure values for visualization
 * This creates a 2D slice of the 3D pressure field for rendering
 */
__global__ void samplePressureSliceKernel(float* output, const float* pressure, 
                                        const AirCavityKernelParams params,
                                        int sliceType, int sliceIndex);

/**
 * Kernel to update visualization vertices for rendering pressure field
 * This is used for CUDA-GL interop visualization
 */
__global__ void updateVisualizationVerticesKernel(float4* vertices, float* colors,
                                                const float* pressure,
                                                const AirCavityKernelParams params,
                                                int sliceType, int sliceIndex,
                                                float colorScale);

/**
 * Calculate a 1D index from 3D coordinates for the pressure field
 * 
 * @param x X coordinate
 * @param y Y coordinate
 * @param z Z coordinate
 * @param width Width of the data array
 * @param height Height of the data array
 * @return 1D index
 */
__device__ __forceinline__ int getPressureIndex(int x, int y, int z, int width, int height) {
    return z * (width * height) + y * width + x;
}

/**
 * Calculate a 1D index from 3D coordinates for the X velocity field
 * X velocity field has dimensions (width+1) x height x depth
 */
__device__ __forceinline__ int getVelocityXIndex(int x, int y, int z, int width, int height) {
    return z * ((width + 1) * height) + y * (width + 1) + x;
}

/**
 * Calculate a 1D index from 3D coordinates for the Y velocity field
 * Y velocity field has dimensions width x (height+1) x depth
 */
__device__ __forceinline__ int getVelocityYIndex(int x, int y, int z, int width, int height) {
    return z * (width * (height + 1)) + y * width + x;
}

/**
 * Calculate a 1D index from 3D coordinates for the Z velocity field
 * Z velocity field has dimensions width x height x (depth+1)
 */
__device__ __forceinline__ int getVelocityZIndex(int x, int y, int z, int width, int height) {
    return z * (width * height) + y * width + x;
}

/**
 * Check if a point is inside the 3D grid boundaries
 */
__device__ __forceinline__ bool isInsideGrid(int x, int y, int z, int width, int height, int depth) {
    return x >= 0 && x < width && y >= 0 && y < height && z >= 0 && z < depth;
}

// Host-side function to launch the initialization kernel
void initializeFields(float* d_pressure, float* d_prevPressure, 
                     float* d_velocityX, float* d_velocityY, float* d_velocityZ,
                     const AirCavityKernelParams& params);

// Host-side function to launch the velocity update kernel
void updateVelocity(float* d_pressure, 
                   float* d_velocityX, float* d_velocityY, float* d_velocityZ,
                   const AirCavityKernelParams& params, float timestep);

// Host-side function to launch the pressure update kernel
void updatePressure(float* d_pressure, float* d_prevPressure,
                   float* d_velocityX, float* d_velocityY, float* d_velocityZ,
                   const AirCavityKernelParams& params, float timestep);

// Host-side function to launch the boundary conditions kernel
void applyBoundaryConditions(float* d_pressure,
                            float* d_velocityX, float* d_velocityY, float* d_velocityZ,
                            const AirCavityKernelParams& params);

// Host-side function to launch the pressure impulse kernel
void addPressureImpulse(float* d_pressure, 
                       const AirCavityKernelParams& params,
                       float x, float y, float z, 
                       float strength, float radius);

// Host-side function to launch the pressure slice sampling kernel
void samplePressureSlice(float* d_output, const float* d_pressure, 
                        const AirCavityKernelParams& params,
                        int sliceType, int sliceIndex);

// Host-side function to launch the visualization vertices update kernel
void updateVisualizationVertices(float4* d_vertices, float* d_colors,
                               const float* d_pressure,
                               const AirCavityKernelParams& params,
                               int sliceType, int sliceIndex,
                               float colorScale);

// Helper to compute a stable time step for the acoustic simulation
float calculateStableTimestep(const AirCavityKernelParams& params);

// Helper to calculate optimal CUDA thread and block configuration for 3D data
dim3 calculateOptimalBlockSize3D(int dataWidth, int dataHeight, int dataDepth);

// Helper to calculate optimal CUDA thread and block configuration for 2D data
dim3 calculateOptimalBlockSize2D(int dataWidth, int dataHeight);

} // namespace drumforge

#endif // DRUMFORGE_AIR_CAVITY_KERNELS_CUH