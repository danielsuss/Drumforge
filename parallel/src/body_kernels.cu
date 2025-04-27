#include "body_kernels.cuh"
#include "cuda_memory_manager.h"
#include <cmath>
#include <iostream>
#include <algorithm>

namespace drumforge {

//-----------------------------------------------------------------------------
// CUDA Kernel Implementations
//-----------------------------------------------------------------------------

__global__ void initializeModesKernel(ResonantMode* modes, const BodyKernelParams params) {
    // Calculate thread's mode index
    int modeIdx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Check bounds
    if (modeIdx >= params.numModes) {
        return;
    }
    
    // Calculate mode indices using consistent formula
    int n = (modeIdx % params.circumferentialModes) + 1;  // 1, 2, 3, 4
    int m = (modeIdx / params.circumferentialModes) % params.axialModes;  // 0, 1, 2, 3
    
    // Base frequency that increases with mode numbers
    float baseFreq = 100.0f + 80.0f * n + 150.0f * m;
    
    // Apply dimensional scaling:
    
    // Radius effect: larger radius = lower frequency (inverse relationship)
    baseFreq *= (5.0f / max(params.radius, 0.1f));
    
    // Height effect: only affects axial modes (m>0)
    if (m > 0) {
        // Taller drum = lower axial frequencies
        baseFreq *= (0.5f / max(params.height, 0.1f));
    }
    
    // Thickness effect: thicker shell = higher frequency
    baseFreq *= (params.thickness / 0.01f) * 0.2f + 0.8f;  // 20% contribution
    
    // Material effects
    baseFreq *= sqrtf(params.stiffness / max(params.density, 0.01f));
    
    // Debug output for the first few modes
    if (modeIdx < 5) {
        printf("Mode %d (n=%d, m=%d): radius=%.2f, height=%.2f, thickness=%.2f, freq=%.2f Hz\n", 
              modeIdx, n, m, params.radius, params.height, params.thickness, baseFreq);
    }
    
    // Set the mode parameters
    modes[modeIdx].frequency = baseFreq;
    modes[modeIdx].amplitude = 1.0f / (1.0f + 0.3f * n + 0.2f * m);  // Higher modes are quieter
    modes[modeIdx].decay = (0.5f + 0.2f * n + 0.3f * m) / params.damping;  // Higher modes decay faster
    modes[modeIdx].phase = 0.0f;
}

__global__ void resetBodyKernel(float* modeStates, float* modeVelocities, int numModes) {
    // Calculate thread ID
    int modeIdx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Check if this thread is within the valid range
    if (modeIdx >= numModes) {
        return;
    }
    
    // Reset mode state and velocity to zero
    modeStates[modeIdx] = 0.0f;
    modeVelocities[modeIdx] = 0.0f;
}

__global__ void updateBodyModesKernel(float* modeStates, float* modeVelocities,
                                   const ResonantMode* modes, const float* excitation,
                                   int numModes, float timestep) {
    // Calculate thread ID
    int modeIdx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Check if this thread is within the valid range
    if (modeIdx >= numModes) {
        return;
    }
    
    // Get mode parameters
    float frequency = modes[modeIdx].frequency;
    float decay = modes[modeIdx].decay;
    
    // Calculate angular frequency (2π * frequency)
    float omega = 2.0f * 3.14159265359f * frequency;
    
    // Get current state
    float state = modeStates[modeIdx];
    float velocity = modeVelocities[modeIdx];
    
    // Apply excitation as a force
    float force = excitation[modeIdx];
    
    // Calculate damping coefficient
    float damping = 1.0f / max(decay, 0.001f); // Ensure not too small
    
    // Update using damped harmonic oscillator equations
    // a = -ω² * x - 2 * ζ * ω * v + F/m
    float acceleration = force - (omega * omega * state) - (damping * velocity);
    
    // Semi-implicit Euler integration
    float newVelocity = velocity + acceleration * timestep;
    float newState = state + newVelocity * timestep;
    
    // Store updated state and velocity
    modeStates[modeIdx] = newState;
    modeVelocities[modeIdx] = newVelocity;
}

__global__ void exciteModeKernel(float* excitation, int modeIndex, float amount, int numModes) {
    // Calculate thread ID
    int modeIdx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Check if this thread is within the valid range
    if (modeIdx >= numModes) {
        return;
    }
    
    // Apply excitation to the specified mode or all modes
    if (modeIndex == -1 || modeIndex == modeIdx) {
        excitation[modeIdx] = amount;
    }
}

__global__ void exciteAllModesKernel(float* excitation, const float* inputExcitation, int numModes) {
    // Calculate thread ID
    int modeIdx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Check if this thread is within the valid range
    if (modeIdx >= numModes) {
        return;
    }
    
    // Copy the excitation value for this mode
    excitation[modeIdx] = inputExcitation[modeIdx];
}

//-----------------------------------------------------------------------------
// Host-side wrapper functions
//-----------------------------------------------------------------------------

void initializeModes(ResonantMode* d_modes, const BodyKernelParams& params) {
    // Calculate optimal block size
    dim3 blockSize = calculateOptimalBlockSize(params.numModes);
    
    // Calculate grid size (ceiling division)
    dim3 gridSize((params.numModes + blockSize.x - 1) / blockSize.x);
    
    // Launch kernel
    initializeModesKernel<<<gridSize, blockSize>>>(d_modes, params);
    
    // Check for errors
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        throw CudaException("Failed to initialize body modes: " + 
                           std::string(cudaGetErrorString(error)));
    }
}

void resetBody(float* d_modeStates, float* d_modeVelocities, int numModes) {
    // Calculate optimal block size
    dim3 blockSize = calculateOptimalBlockSize(numModes);
    
    // Calculate grid size (ceiling division)
    dim3 gridSize((numModes + blockSize.x - 1) / blockSize.x);
    
    // Launch kernel
    resetBodyKernel<<<gridSize, blockSize>>>(d_modeStates, d_modeVelocities, numModes);
    
    // Check for errors
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        throw CudaException("Failed to reset body: " + 
                           std::string(cudaGetErrorString(error)));
    }
}

void updateBodyModes(float* d_modeStates, float* d_modeVelocities,
                  const ResonantMode* d_modes, const float* d_excitation,
                  int numModes, float timestep) {
    // Calculate optimal block size
    dim3 blockSize = calculateOptimalBlockSize(numModes);
    
    // Calculate grid size (ceiling division)
    dim3 gridSize((numModes + blockSize.x - 1) / blockSize.x);
    
    // Launch kernel
    updateBodyModesKernel<<<gridSize, blockSize>>>(
        d_modeStates, d_modeVelocities, d_modes, d_excitation, numModes, timestep
    );
    
    // Check for errors
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        throw CudaException("Failed to update body modes: " + 
                           std::string(cudaGetErrorString(error)));
    }
}

void exciteMode(float* d_excitation, int modeIndex, float amount, int numModes) {
    // Calculate optimal block size
    dim3 blockSize = calculateOptimalBlockSize(numModes);
    
    // Calculate grid size (ceiling division)
    dim3 gridSize((numModes + blockSize.x - 1) / blockSize.x);
    
    // Launch kernel
    exciteModeKernel<<<gridSize, blockSize>>>(d_excitation, modeIndex, amount, numModes);
    
    // Check for errors
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        throw CudaException("Failed to excite body mode: " + 
                           std::string(cudaGetErrorString(error)));
    }
}

void exciteAllModes(float* d_excitation, const float* d_inputExcitation, int numModes) {
    // Calculate optimal block size
    dim3 blockSize = calculateOptimalBlockSize(numModes);
    
    // Calculate grid size (ceiling division)
    dim3 gridSize((numModes + blockSize.x - 1) / blockSize.x);
    
    // Launch kernel
    exciteAllModesKernel<<<gridSize, blockSize>>>(d_excitation, d_inputExcitation, numModes);
    
    // Check for errors
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        throw CudaException("Failed to excite all body modes: " + 
                           std::string(cudaGetErrorString(error)));
    }
}

dim3 calculateOptimalBlockSize(int dataSize) {
    // Default block size (1D)
    int blockSizeX = 256;
    
    // For very small data sizes, use fewer threads
    if (dataSize < 256) {
        blockSizeX = std::max(32, (dataSize + 31) / 32 * 32);
    }
    
    return dim3(blockSizeX);
}

void setupMaterialPreset(BodyKernelParams& params, const std::string& material) {
    std::cout << "Setting up material preset: " << material << std::endl;
    
    // Set default values (Maple)
    params.density = 1.0f;
    params.stiffness = 1.0f;
    params.damping = 0.05f;
    
    // Adjust based on material type
    if (material == "Maple") {
        // Maple (medium-dense hardwood with balanced tone)
        params.density = 1.0f;
        params.stiffness = 1.0f;
        params.damping = 0.05f;
    }
    else if (material == "Birch") {
        // Birch (brighter sound with more attack)
        params.density = 0.9f;
        params.stiffness = 1.2f;
        params.damping = 0.04f;
    }
    else if (material == "Mahogany") {
        // Mahogany (warm and deep sound)
        params.density = 0.8f;
        params.stiffness = 0.8f;
        params.damping = 0.06f;
    }
    else if (material == "Metal") {
        // Metal (bright with long sustain)
        params.density = 4.0f;
        params.stiffness = 7.0f;
        params.damping = 0.01f;
    }
    else if (material == "Acrylic") {
        // Acrylic (clear sound with less resonance)
        params.density = 1.7f;
        params.stiffness = 0.4f;
        params.damping = 0.08f;
    }
    
    std::cout << "Material properties set: density=" << params.density
              << ", stiffness=" << params.stiffness
              << ", damping=" << params.damping << std::endl;
}

float calculateStableTimestep(const BodyKernelParams& params) {
    // For modal synthesis, the timestep should be small enough to
    // accurately sample the highest frequency mode
    float maxFreq = 0.0f;
    
    // Approximate the highest possible frequency based on mode parameters
    // This is a rough estimate based on our frequency calculation
    int maxN = params.circumferentialModes;
    int maxM = params.axialModes - 1;
    
    float baseFreq = 100.0f + 80.0f * maxN + 150.0f * maxM;
    
    // Apply material scaling
    float materialFactor = std::sqrt(params.stiffness / std::max(params.density, 0.01f));
    maxFreq = baseFreq * materialFactor;
    
    // Ensure we have enough samples per cycle (at least 10)
    float minTimestep = 1.0f / (maxFreq * 10.0f);
    
    // Add safety factor
    return minTimestep * 0.8f;
}

} // namespace drumforge