#include "air_cavity_kernels.cuh"
#include "cuda_memory_manager.h"
#include <cmath>
#include <iostream>

namespace drumforge {

//-----------------------------------------------------------------------------
// CUDA Kernel Implementations
//-----------------------------------------------------------------------------

__global__ void initializeFieldsKernel(float* pressure, float* prevPressure, 
                                    float* velocityX, float* velocityY, float* velocityZ,
                                    const AirCavityKernelParams params) {
    // Calculate thread's position in the 3D grid
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;
    
    // Check bounds for pressure fields
    if (x < params.cavityWidth && y < params.cavityHeight && z < params.cavityDepth) {
        // Get linear index into pressure arrays
        int pressureIdx = getPressureIndex(x, y, z, params.cavityWidth, params.cavityHeight);
        
        // Initialize pressure fields to zero
        pressure[pressureIdx] = 0.0f;
        prevPressure[pressureIdx] = 0.0f;
    }
    
    // Check bounds for velocity fields (staggered grid)
    // X velocity has dimensions (width+1) x height x depth
    if (x < params.cavityWidth + 1 && y < params.cavityHeight && z < params.cavityDepth) {
        int vxIdx = getVelocityXIndex(x, y, z, params.cavityWidth, params.cavityHeight);
        velocityX[vxIdx] = 0.0f;
    }
    
    // Y velocity has dimensions width x (height+1) x depth
    if (x < params.cavityWidth && y < params.cavityHeight + 1 && z < params.cavityDepth) {
        int vyIdx = getVelocityYIndex(x, y, z, params.cavityWidth, params.cavityHeight);
        velocityY[vyIdx] = 0.0f;
    }
    
    // Z velocity has dimensions width x height x (depth+1)
    if (x < params.cavityWidth && y < params.cavityHeight && z < params.cavityDepth + 1) {
        int vzIdx = getVelocityZIndex(x, y, z, params.cavityWidth, params.cavityHeight);
        velocityZ[vzIdx] = 0.0f;
    }
}

__global__ void updateVelocityKernel(float* pressure, 
                                   float* velocityX, float* velocityY, float* velocityZ,
                                   const AirCavityKernelParams params, float timestep) {
    // Calculate thread's position in the 3D grid
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;
    
    // Width, height and depth for convenience
    int width = params.cavityWidth;
    int height = params.cavityHeight;
    int depth = params.cavityDepth;
    
    // Physical constants for the simulation
    float dt = timestep;                    // Time step
    float dx = params.cellSize;             // Grid cell size
    float rho = params.density;             // Air density
    
    // Coefficient for pressure gradient in momentum equation
    float coef = -dt / (rho * dx);
    
    // Update X velocity
    // Velocity X is defined at (i+1/2, j, k)
    if (x < width && y < height && z < depth) {
        int vxIdx = getVelocityXIndex(x, y, z, width, height);
        
        // Get pressure at (i, j, k) and (i+1, j, k)
        int p1Idx = getPressureIndex(x, y, z, width, height);
        
        // Check if we're at the right edge
        if (x + 1 < width) {
            int p2Idx = getPressureIndex(x + 1, y, z, width, height);
            
            // Calculate pressure gradient in x direction
            float gradP = (pressure[p2Idx] - pressure[p1Idx]) / dx;
            
            // Update velocity using momentum equation
            velocityX[vxIdx] += coef * gradP;
            
            // Apply damping
            velocityX[vxIdx] *= (1.0f - params.dampingCoefficient);
        }
    }
    
    // Update Y velocity
    // Velocity Y is defined at (i, j+1/2, k)
    if (x < width && y < height && z < depth) {
        int vyIdx = getVelocityYIndex(x, y, z, width, height);
        
        // Get pressure at (i, j, k) and (i, j+1, k)
        int p1Idx = getPressureIndex(x, y, z, width, height);
        
        // Check if we're at the top edge
        if (y + 1 < height) {
            int p2Idx = getPressureIndex(x, y + 1, z, width, height);
            
            // Calculate pressure gradient in y direction
            float gradP = (pressure[p2Idx] - pressure[p1Idx]) / dx;
            
            // Update velocity using momentum equation
            velocityY[vyIdx] += coef * gradP;
            
            // Apply damping
            velocityY[vyIdx] *= (1.0f - params.dampingCoefficient);
        }
    }
    
    // Update Z velocity
    // Velocity Z is defined at (i, j, k+1/2)
    if (x < width && y < height && z < depth) {
        int vzIdx = getVelocityZIndex(x, y, z, width, height);
        
        // Get pressure at (i, j, k) and (i, j, k+1)
        int p1Idx = getPressureIndex(x, y, z, width, height);
        
        // Check if we're at the front edge
        if (z + 1 < depth) {
            int p2Idx = getPressureIndex(x, y, z + 1, width, height);
            
            // Calculate pressure gradient in z direction
            float gradP = (pressure[p2Idx] - pressure[p1Idx]) / dx;
            
            // Update velocity using momentum equation
            velocityZ[vzIdx] += coef * gradP;
            
            // Apply damping
            velocityZ[vzIdx] *= (1.0f - params.dampingCoefficient);
        }
    }
}

__global__ void updatePressureKernel(float* pressure, float* prevPressure,
                                   float* velocityX, float* velocityY, float* velocityZ,
                                   const AirCavityKernelParams params, float timestep) {
    // Calculate thread's position in the 3D grid
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;
    
    // Width, height and depth for convenience
    int width = params.cavityWidth;
    int height = params.cavityHeight;
    int depth = params.cavityDepth;
    
    // Skip if outside domain
    if (x >= width || y >= height || z >= depth) {
        return;
    }
    
    // Skip boundary cells (will be handled by boundary conditions)
    if (x == 0 || x == width - 1 || y == 0 || y == height - 1 || z == 0 || z == depth - 1) {
        return;
    }
    
    // Physical constants
    float dt = timestep;                    // Time step
    float dx = params.cellSize;             // Grid cell size
    float c = params.speedOfSound;          // Speed of sound in air
    float rho = params.density;             // Air density
    float K = rho * c * c;                  // Bulk modulus of air (rho * c^2)
    
    // Get the pressure index for this cell
    int pIdx = getPressureIndex(x, y, z, width, height);
    
    // Save previous pressure
    prevPressure[pIdx] = pressure[pIdx];
    
    // Calculate divergence of velocity field
    // Get velocity values for staggered grid
    int vxLeftIdx = getVelocityXIndex(x - 1, y, z, width, height);
    int vxRightIdx = getVelocityXIndex(x, y, z, width, height);
    
    int vyBottomIdx = getVelocityYIndex(x, y - 1, z, width, height);
    int vyTopIdx = getVelocityYIndex(x, y, z, width, height);
    
    int vzBackIdx = getVelocityZIndex(x, y, z - 1, width, height);
    int vzFrontIdx = getVelocityZIndex(x, y, z, width, height);
    
    // Calculate velocity divergence
    float divV = (velocityX[vxRightIdx] - velocityX[vxLeftIdx] +
                 velocityY[vyTopIdx] - velocityY[vyBottomIdx] +
                 velocityZ[vzFrontIdx] - velocityZ[vzBackIdx]) / dx;
    
    // Update pressure using the acoustic wave equation (linearized Euler equations)
    pressure[pIdx] = prevPressure[pIdx] - K * dt * divV;
    
    // Apply damping
    pressure[pIdx] *= (1.0f - params.dampingCoefficient);
}

__global__ void applyBoundaryConditionsKernel(float* pressure,
                                           float* velocityX, float* velocityY, float* velocityZ,
                                           const AirCavityKernelParams params) {
    // Calculate thread's position in the 3D grid
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;
    
    // Width, height and depth for convenience
    int width = params.cavityWidth;
    int height = params.cavityHeight;
    int depth = params.cavityDepth;
    
    // Apply boundary conditions to pressure field
    if (x < width && y < height && z < depth) {
        // Skip interior cells
        if (x > 0 && x < width - 1 && y > 0 && y < height - 1 && z > 0 && z < depth - 1) {
            return;
        }
        
        int pIdx = getPressureIndex(x, y, z, width, height);
        
        // For simplicity, implementing reflective boundary conditions
        // This means setting the pressure at the boundary to mirror the pressure just inside
        
        // Left boundary (x = 0)
        if (x == 0) {
            int mirrorIdx = getPressureIndex(1, y, z, width, height);
            pressure[pIdx] = pressure[mirrorIdx];
        }
        
        // Right boundary (x = width - 1)
        if (x == width - 1) {
            int mirrorIdx = getPressureIndex(width - 2, y, z, width, height);
            pressure[pIdx] = pressure[mirrorIdx];
        }
        
        // Bottom boundary (y = 0)
        if (y == 0) {
            int mirrorIdx = getPressureIndex(x, 1, z, width, height);
            pressure[pIdx] = pressure[mirrorIdx];
        }
        
        // Top boundary (y = height - 1)
        if (y == height - 1) {
            int mirrorIdx = getPressureIndex(x, height - 2, z, width, height);
            pressure[pIdx] = pressure[mirrorIdx];
        }
        
        // Back boundary (z = 0)
        if (z == 0) {
            int mirrorIdx = getPressureIndex(x, y, 1, width, height);
            pressure[pIdx] = pressure[mirrorIdx];
        }
        
        // Front boundary (z = depth - 1)
        if (z == depth - 1) {
            int mirrorIdx = getPressureIndex(x, y, depth - 2, width, height);
            pressure[pIdx] = pressure[mirrorIdx];
        }
    }
    
    // Apply boundary conditions to velocity fields
    // For reflective boundaries, normal velocity is zero at the boundary
    
    // X velocity boundaries
    if (x <= width && y < height && z < depth) {
        // Left and right boundaries (x = 0 and x = width)
        if (x == 0 || x == width) {
            int vxIdx = getVelocityXIndex(x, y, z, width, height);
            velocityX[vxIdx] = 0.0f;
        }
    }
    
    // Y velocity boundaries
    if (x < width && y <= height && z < depth) {
        // Bottom and top boundaries (y = 0 and y = height)
        if (y == 0 || y == height) {
            int vyIdx = getVelocityYIndex(x, y, z, width, height);
            velocityY[vyIdx] = 0.0f;
        }
    }
    
    // Z velocity boundaries
    if (x < width && y < height && z <= depth) {
        // Back and front boundaries (z = 0 and z = depth)
        if (z == 0 || z == depth) {
            int vzIdx = getVelocityZIndex(x, y, z, width, height);
            velocityZ[vzIdx] = 0.0f;
        }
    }
}

__global__ void addPressureImpulseKernel(float* pressure, 
                                       const AirCavityKernelParams params,
                                       float x, float y, float z, 
                                       float strength, float radius) {
    // Calculate thread's position in the 3D grid
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;
    int iz = blockIdx.z * blockDim.z + threadIdx.z;
    
    // Width, height and depth for convenience
    int width = params.cavityWidth;
    int height = params.cavityHeight;
    int depth = params.cavityDepth;
    
    // Skip if outside domain
    if (ix >= width || iy >= height || iz >= depth) {
        return;
    }
    
    // Convert normalized or world position to grid indices if necessary
    int centerX = (int)(x * width);
    int centerY = (int)(y * height);
    int centerZ = (int)(z * depth);
    
    // If x, y, z are already in grid coordinates (not normalized), use them directly
    if (x >= 1.0f) centerX = (int)x;
    if (y >= 1.0f) centerY = (int)y;
    if (z >= 1.0f) centerZ = (int)z;
    
    // Calculate distance squared from impulse center
    float dx = ix - centerX;
    float dy = iy - centerY;
    float dz = iz - centerZ;
    float distSquared = dx*dx + dy*dy + dz*dz;
    
    // Get radius squared
    float radiusSquared = radius * radius;
    
    // Only apply impulse within the radius
    if (distSquared <= radiusSquared) {
        // Calculate pressure index
        int pIdx = getPressureIndex(ix, iy, iz, width, height);
        
        // Apply a Gaussian-shaped impulse
        float dist = sqrtf(distSquared);
        float falloff = expf(-dist*dist / (radiusSquared/4.0f));
        
        // Apply the impulse
        pressure[pIdx] += strength * falloff;
    }
}

__global__ void samplePressureSliceKernel(float* output, const float* pressure, 
                                        const AirCavityKernelParams params,
                                        int sliceType, int sliceIndex) {
    // Calculate thread's position in the 2D slice
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    // Width, height and depth for convenience
    int width = params.cavityWidth;
    int height = params.cavityHeight;
    int depth = params.cavityDepth;
    
    // Skip if outside domain
    bool outOfBounds = false;
    int pressureIdx = 0;
    int outputIdx = 0;
    
    // Process based on slice type
    switch (sliceType) {
        case 0: // XY slice (constant z)
            // Bound check slice index
            if (sliceIndex < 0 || sliceIndex >= depth) {
                outOfBounds = true;
                break;
            }
            
            // Bound check x, y
            if (x >= width || y >= height) {
                outOfBounds = true;
                break;
            }
            
            // Calculate indices
            pressureIdx = getPressureIndex(x, y, sliceIndex, width, height);
            outputIdx = y * width + x;
            break;
            
        case 1: // XZ slice (constant y)
            // Bound check slice index
            if (sliceIndex < 0 || sliceIndex >= height) {
                outOfBounds = true;
                break;
            }
            
            // Bound check x, y (where y is used as z in this context)
            if (x >= width || y >= depth) {
                outOfBounds = true;
                break;
            }
            
            // Calculate indices
            pressureIdx = getPressureIndex(x, sliceIndex, y, width, height);
            outputIdx = y * width + x;
            break;
            
        case 2: // YZ slice (constant x)
            // Bound check slice index
            if (sliceIndex < 0 || sliceIndex >= width) {
                outOfBounds = true;
                break;
            }
            
            // Bound check x, y (where x is used as y and y as z in this context)
            if (x >= height || y >= depth) {
                outOfBounds = true;
                break;
            }
            
            // Calculate indices
            pressureIdx = getPressureIndex(sliceIndex, x, y, width, height);
            outputIdx = y * height + x;
            break;
            
        default:
            outOfBounds = true;
            break;
    }
    
    // If within bounds, copy pressure value to output
    if (!outOfBounds) {
        output[outputIdx] = pressure[pressureIdx];
    }
}

__global__ void updateVisualizationVerticesKernel(float4* vertices, float* colors,
                                                const float* pressure,
                                                const AirCavityKernelParams params,
                                                int sliceType, int sliceIndex,
                                                float colorScale) {
    // Calculate thread's position in the 2D slice
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    // Width, height and depth for convenience
    int width = params.cavityWidth;
    int height = params.cavityHeight;
    int depth = params.cavityDepth;
    
    // Cell size for world positioning
    float cellSize = params.cellSize;
    
    // Skip if outside domain
    bool outOfBounds = false;
    int pressureIdx = 0;
    int vertexIdx = 0;
    
    // World space dimensions and position calculations
    float worldX = 0.0f, worldY = 0.0f, worldZ = 0.0f;
    int dimX = 0, dimY = 0;
    
    // Process based on slice type
    switch (sliceType) {
        case 0: // XY slice (constant z)
            if (sliceIndex < 0 || sliceIndex >= depth || x >= width || y >= height) {
                outOfBounds = true;
                break;
            }
            
            pressureIdx = getPressureIndex(x, y, sliceIndex, width, height);
            vertexIdx = y * width + x;
            
            worldX = (x - width/2.0f) * cellSize;
            worldY = (y - height/2.0f) * cellSize;
            worldZ = (sliceIndex - depth/2.0f) * cellSize;
            
            dimX = width;
            dimY = height;
            break;
            
        case 1: // XZ slice (constant y)
            if (sliceIndex < 0 || sliceIndex >= height || x >= width || y >= depth) {
                outOfBounds = true;
                break;
            }
            
            pressureIdx = getPressureIndex(x, sliceIndex, y, width, height);
            vertexIdx = y * width + x;
            
            worldX = (x - width/2.0f) * cellSize;
            worldY = (sliceIndex - height/2.0f) * cellSize;
            worldZ = (y - depth/2.0f) * cellSize;
            
            dimX = width;
            dimY = depth;
            break;
            
        case 2: // YZ slice (constant x)
            if (sliceIndex < 0 || sliceIndex >= width || x >= height || y >= depth) {
                outOfBounds = true;
                break;
            }
            
            pressureIdx = getPressureIndex(sliceIndex, x, y, width, height);
            vertexIdx = y * height + x;
            
            worldX = (sliceIndex - width/2.0f) * cellSize;
            worldY = (x - height/2.0f) * cellSize;
            worldZ = (y - depth/2.0f) * cellSize;
            
            dimX = height;
            dimY = depth;
            break;
            
        default:
            outOfBounds = true;
            break;
    }
    
    // If within bounds, update vertex and color
    if (!outOfBounds) {
        // Get pressure value
        float pressureValue = pressure[pressureIdx];
        
        // Scale pressure for visualization
        float displacement = pressureValue * colorScale;
        
        // Create vertex with position
        vertices[vertexIdx] = make_float4(worldX, worldY, worldZ + displacement, 1.0f);
        
        // Create color based on pressure
        // Red for positive pressure, blue for negative
        if (displacement > 0.0f) {
            // Positive pressure: red to yellow
            float intensity = fminf(fabsf(displacement), 1.0f);
            colors[vertexIdx*4] = 1.0f; // R
            colors[vertexIdx*4+1] = intensity; // G
            colors[vertexIdx*4+2] = 0.0f; // B
            colors[vertexIdx*4+3] = 1.0f; // A
        } else {
            // Negative pressure: blue to cyan
            float intensity = fminf(fabsf(displacement), 1.0f);
            colors[vertexIdx*4] = 0.0f; // R
            colors[vertexIdx*4+1] = intensity; // G
            colors[vertexIdx*4+2] = 1.0f; // B
            colors[vertexIdx*4+3] = 1.0f; // A
        }
    }
}

//-----------------------------------------------------------------------------
// Host-side wrapper functions
//-----------------------------------------------------------------------------

void initializeFields(float* d_pressure, float* d_prevPressure, 
                     float* d_velocityX, float* d_velocityY, float* d_velocityZ,
                     const AirCavityKernelParams& params) {
    // Calculate optimal block size
    dim3 blockSize = calculateOptimalBlockSize3D(params.cavityWidth, params.cavityHeight, params.cavityDepth);
    
    // Calculate grid size
    dim3 gridSize(
        (params.cavityWidth + blockSize.x - 1) / blockSize.x,
        (params.cavityHeight + blockSize.y - 1) / blockSize.y,
        (params.cavityDepth + blockSize.z - 1) / blockSize.z
    );
    
    // Launch kernel
    initializeFieldsKernel<<<gridSize, blockSize>>>(
        d_pressure, d_prevPressure, d_velocityX, d_velocityY, d_velocityZ, params
    );
    
    // Check for errors
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        throw CudaException("Failed to initialize air cavity fields: " + 
                           std::string(cudaGetErrorString(error)));
    }
    
    // Synchronize to ensure initialization is complete
    cudaDeviceSynchronize();
}

void updateVisualizationVertices(float4* d_vertices, float* d_colors,
                               const float* d_pressure,
                               const AirCavityKernelParams& params,
                               int sliceType, int sliceIndex,
                               float colorScale) {
    // Determine dimensions based on slice type
    int dimX = 0, dimY = 0;
    
    switch (sliceType) {
        case 0: // XY slice
            dimX = params.cavityWidth;
            dimY = params.cavityHeight;
            break;
        case 1: // XZ slice
            dimX = params.cavityWidth;
            dimY = params.cavityDepth;
            break;
        case 2: // YZ slice
            dimX = params.cavityHeight;
            dimY = params.cavityDepth;
            break;
        default:
            throw std::invalid_argument("Invalid slice type");
    }
    
    // Calculate optimal block size for 2D
    dim3 blockSize = calculateOptimalBlockSize2D(dimX, dimY);
    
    // Calculate grid size
    dim3 gridSize(
        (dimX + blockSize.x - 1) / blockSize.x,
        (dimY + blockSize.y - 1) / blockSize.y
    );
    
    // Launch kernel
    updateVisualizationVerticesKernel<<<gridSize, blockSize>>>(
        d_vertices, d_colors, d_pressure, params, sliceType, sliceIndex, colorScale
    );
    
    // Check for errors
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        throw CudaException("Failed to update visualization vertices: " + 
                           std::string(cudaGetErrorString(error)));
    }
}

void updateVelocity(float* d_pressure, 
                   float* d_velocityX, float* d_velocityY, float* d_velocityZ,
                   const AirCavityKernelParams& params, float timestep) {
    // Calculate optimal block size
    dim3 blockSize = calculateOptimalBlockSize3D(params.cavityWidth, params.cavityHeight, params.cavityDepth);
    
    // Calculate grid size
    dim3 gridSize(
        (params.cavityWidth + blockSize.x - 1) / blockSize.x,
        (params.cavityHeight + blockSize.y - 1) / blockSize.y,
        (params.cavityDepth + blockSize.z - 1) / blockSize.z
    );
    
    // Launch kernel
    updateVelocityKernel<<<gridSize, blockSize>>>(
        d_pressure, d_velocityX, d_velocityY, d_velocityZ, params, timestep
    );
    
    // Check for errors
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        throw CudaException("Failed to update air cavity velocity: " + 
                           std::string(cudaGetErrorString(error)));
    }
}

void updatePressure(float* d_pressure, float* d_prevPressure,
                   float* d_velocityX, float* d_velocityY, float* d_velocityZ,
                   const AirCavityKernelParams& params, float timestep) {
    // Calculate optimal block size
    dim3 blockSize = calculateOptimalBlockSize3D(params.cavityWidth, params.cavityHeight, params.cavityDepth);
    
    // Calculate grid size
    dim3 gridSize(
        (params.cavityWidth + blockSize.x - 1) / blockSize.x,
        (params.cavityHeight + blockSize.y - 1) / blockSize.y,
        (params.cavityDepth + blockSize.z - 1) / blockSize.z
    );
    
    // Launch kernel
    updatePressureKernel<<<gridSize, blockSize>>>(
        d_pressure, d_prevPressure, d_velocityX, d_velocityY, d_velocityZ, params, timestep
    );
    
    // Check for errors
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        throw CudaException("Failed to update air cavity pressure: " + 
                           std::string(cudaGetErrorString(error)));
    }
}

void applyBoundaryConditions(float* d_pressure,
                           float* d_velocityX, float* d_velocityY, float* d_velocityZ,
                           const AirCavityKernelParams& params) {
    // Calculate optimal block size
    dim3 blockSize = calculateOptimalBlockSize3D(params.cavityWidth, params.cavityHeight, params.cavityDepth);
    
    // Calculate grid size
    dim3 gridSize(
        (params.cavityWidth + blockSize.x - 1) / blockSize.x,
        (params.cavityHeight + blockSize.y - 1) / blockSize.y,
        (params.cavityDepth + blockSize.z - 1) / blockSize.z
    );
    
    // Launch kernel
    applyBoundaryConditionsKernel<<<gridSize, blockSize>>>(
        d_pressure, d_velocityX, d_velocityY, d_velocityZ, params
    );
    
    // Check for errors
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        throw CudaException("Failed to apply air cavity boundary conditions: " + 
                           std::string(cudaGetErrorString(error)));
    }
}

void addPressureImpulse(float* d_pressure, 
                       const AirCavityKernelParams& params,
                       float x, float y, float z, 
                       float strength, float radius) {
    // Calculate optimal block size
    dim3 blockSize = calculateOptimalBlockSize3D(params.cavityWidth, params.cavityHeight, params.cavityDepth);
    
    // Calculate grid size
    dim3 gridSize(
        (params.cavityWidth + blockSize.x - 1) / blockSize.x,
        (params.cavityHeight + blockSize.y - 1) / blockSize.y,
        (params.cavityDepth + blockSize.z - 1) / blockSize.z
    );
    
    // Launch kernel
    addPressureImpulseKernel<<<gridSize, blockSize>>>(
        d_pressure, params, x, y, z, strength, radius
    );
    
    // Check for errors
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        throw CudaException("Failed to add pressure impulse: " + 
                           std::string(cudaGetErrorString(error)));
    }
}

void samplePressureSlice(float* d_output, const float* d_pressure, 
                        const AirCavityKernelParams& params,
                        int sliceType, int sliceIndex) {
    // Determine dimensions based on slice type
    int dimX = 0, dimY = 0;
    
    switch (sliceType) {
        case 0: // XY slice
            dimX = params.cavityWidth;
            dimY = params.cavityHeight;
            break;
        case 1: // XZ slice
            dimX = params.cavityWidth;
            dimY = params.cavityDepth;
            break;
        case 2: // YZ slice
            dimX = params.cavityHeight;
            dimY = params.cavityDepth;
            break;
        default:
            throw std::invalid_argument("Invalid slice type");
    }
    
    // Calculate optimal block size for 2D
    dim3 blockSize = calculateOptimalBlockSize2D(dimX, dimY);
    
    // Calculate grid size
    dim3 gridSize(
        (dimX + blockSize.x - 1) / blockSize.x,
        (dimY + blockSize.y - 1) / blockSize.y
    );
    
    // Launch kernel
    samplePressureSliceKernel<<<gridSize, blockSize>>>(
        d_output, d_pressure, params, sliceType, sliceIndex
    );
    
    // Check for errors
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        throw CudaException("Failed to sample pressure slice: " + 
                           std::string(cudaGetErrorString(error)));
    }
}

}