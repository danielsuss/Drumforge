#include "body_kernels.cuh"
#include "body_component.h"
#include "cuda_memory_manager.h"
#include <cmath>
#include <iostream>
#include <algorithm>

namespace drumforge {

//-----------------------------------------------------------------------------
// CUDA Kernel Implementations
//-----------------------------------------------------------------------------

__global__ void initializeModesKernel(ResonantMode* modes, const BodyKernelParams params) {
    // Calculate thread ID
    int modeIdx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Check if this thread is within the valid range
    if (modeIdx >= params.numModes) {
        return;
    }
    
    // Calculate normalized position in the frequency range [0,1]
    float normalizedPos;
    if (params.modeSpacing == 1.0f) {
        // Linear frequency spacing
        normalizedPos = static_cast<float>(modeIdx) / (params.numModes - 1);
    } else {
        // Stretched frequency spacing (higher modes more spread out)
        normalizedPos = powf(static_cast<float>(modeIdx) / (params.numModes - 1), params.modeSpacing);
    }
    
    // Calculate frequency using logarithmic scaling between min and max frequency
    float logMinFreq = logf(params.minFrequency);
    float logMaxFreq = logf(params.maxFrequency);
    float frequency = expf(logMinFreq + normalizedPos * (logMaxFreq - logMinFreq));
    
    // Calculate amplitude (typically decreases with frequency)
    // A simple model: amplitude ~ 1/sqrt(frequency)
    float amplitude = 1.0f / sqrtf(frequency / params.minFrequency);
    
    // Scale amplitude by material factors
    amplitude *= params.thickness;  // Thicker shells have more amplitude
    amplitude /= params.density;    // Denser materials have less amplitude
    
    // Calculate decay time (higher frequencies decay faster)
    // A typical model: decay ~ 1/frequency
    float decay = params.dampingFactor * (params.minFrequency / frequency);
    
    // Scale decay by material properties
    decay *= params.thickness;      // Thicker shells decay slower
    decay *= sqrtf(params.density); // Denser materials decay slower
    
    // Set initial phase to zero
    float phase = 0.0f;
    
    // Set the mode parameters
    modes[modeIdx].frequency = frequency;
    modes[modeIdx].amplitude = amplitude;
    modes[modeIdx].decay = decay;
    modes[modeIdx].phase = phase;
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
    float damping = 1.0f / (decay * frequency + 0.0001f); // Avoid division by zero
    
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

__host__ void setupMaterialPreset(BodyKernelParams& params, const std::string& material) {
    // Default values for maple (medium density hardwood)
    params.density = 0.7f;
    params.youngsModulus = 1.0f;
    params.dampingFactor = 0.02f;
    
    // Set parameters based on material type
    if (material == "Maple") {
        // Maple is the default (medium-dense hardwood with balanced tone)
        params.density = 0.7f;
        params.youngsModulus = 1.0f;
        params.dampingFactor = 0.02f;
        params.minFrequency = 80.0f;
        params.maxFrequency = 4000.0f;
        params.modeSpacing = 1.1f;
    }
    else if (material == "Birch") {
        // Birch (brighter sound with more attack)
        params.density = 0.65f;
        params.youngsModulus = 1.1f;
        params.dampingFactor = 0.015f;
        params.minFrequency = 100.0f;
        params.maxFrequency = 5000.0f;
        params.modeSpacing = 1.15f;
    }
    else if (material == "Mahogany") {
        // Mahogany (warm and deep sound)
        params.density = 0.55f;
        params.youngsModulus = 0.9f;
        params.dampingFactor = 0.025f;
        params.minFrequency = 60.0f;
        params.maxFrequency = 3500.0f;
        params.modeSpacing = 1.05f;
    }
    else if (material == "Metal") {
        // Metal (bright with long sustain)
        params.density = 1.2f;
        params.youngsModulus = 2.0f;
        params.dampingFactor = 0.005f;
        params.minFrequency = 120.0f;
        params.maxFrequency = 8000.0f;
        params.modeSpacing = 1.2f;
    }
    else if (material == "Acrylic") {
        // Acrylic (clear sound with less resonance)
        params.density = 0.8f;
        params.youngsModulus = 0.7f;
        params.dampingFactor = 0.04f;
        params.minFrequency = 100.0f;
        params.maxFrequency = 3000.0f;
        params.modeSpacing = 1.0f;
    }
    else {
        // Unknown material, use maple as default
        std::cerr << "Unknown material '" << material 
                  << "', using Maple as default" << std::endl;
    }
}

float calculateStableTimestep(const BodyKernelParams& params) {
    // KEY OPTIMIZATION: For physics, we only need stability up to ~1000Hz to maintain
    // visual and physical accuracy. Higher frequencies are important for audio
    // but don't need precise physics integration
    float physicsMaxFreq = std::min(params.maxFrequency, 1000.0f);
    
    // The stable timestep is determined by the highest frequency
    // A rule of thumb is: dt <= 1/(π*fmax)
    float stableTimestep = 1.0f / (3.14159265359f * physicsMaxFreq);
    
    // Apply safety factor (0.8) for numerical stability
    return 0.8f * stableTimestep;
}

} // namespace drumforge