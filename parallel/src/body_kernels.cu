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
    // Calculate thread's mode index
    int modeIdx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Check bounds
    if (modeIdx >= params.numModes) {
        return;
    }
    
    // Determine circumferential (n) and axial (m) mode numbers
    // We'll distribute the modes to get a good variety of both types
    int n = modeIdx % params.maxCircumferentialModes;  // 0, 1, 2, ... around circumference
    int m = (modeIdx / params.maxCircumferentialModes) % params.maxAxialModes;  // 0, 1, 2, ... along height
    
    // Avoid n=0, m=0 mode (rigid body mode with zero frequency)
    if (n == 0 && m == 0) {
        n = 1;  // Use n=1 instead
    }
    
    // Calculate frequency using the Donnell-Mushtari shell theory formula
    // f(m,n) = (1/2π) * √(E/[ρ(1-ν²)]) * √[(h²/12R²) * [n² + (mπR/L)²]² + (1/R²) * [1 + (mπR/L)²]]
    
    // Material term
    float materialTerm = sqrtf(params.youngsModulus / 
                              (params.density * (1.0f - params.poissonsRatio * params.poissonsRatio)));
    
    // Geometry terms
    float R = params.radius;
    float L = params.height;
    float h = params.thickness;
    float nTerm = (float)n;
    float mTerm = (float)m * M_PI * R / L;
    
    // Bending term
    float bendingTerm = (h*h)/(12.0f*R*R) * powf(nTerm*nTerm + mTerm*mTerm, 2.0f);
    
    // Membrane term
    float membraneTerm = (1.0f/(R*R)) * (1.0f + mTerm*mTerm);
    
    // Combined frequency calculation
    float frequency = (1.0f/(2.0f*M_PI)) * materialTerm * sqrtf(bendingTerm + membraneTerm);
    
    // Scale frequency to a reasonable range to prevent extreme values
    // The physical formula gives raw frequencies that may need adjustment
    // to work well in our audio system
    float scaleFactor = 500.0f;  // Adjust based on testing
    frequency = frequency * scaleFactor;
    
    // Ensure frequency is within our simulation's limits
    frequency = fmaxf(params.minFrequency, fminf(frequency, params.maxFrequency));
    
    // Calculate relative amplitude based on mode
    // Lower modes typically have higher amplitude in real drums
    float amplitudeFactor = 1.0f / sqrtf(1.0f + 0.5f*n + 0.3f*m);
    
    // Calculate decay time - higher modes decay faster
    float decayFactor = params.dampingFactor * (0.1f + 0.4f*n + 0.3f*m);
    
    // Material-specific adjustments
    // Wood has more damping in higher modes
    if (params.youngsModulus < 20.0f) {  // Wood materials
        decayFactor *= (1.0f + 0.5f*n*n);
    }
    
    // Adjust for thickness
    amplitudeFactor *= sqrtf(params.thickness);  // Thicker shells have more amplitude
    decayFactor *= (1.0f / params.thickness);    // Thicker shells decay slower
    
    // Set the mode parameters
    modes[modeIdx].frequency = frequency;
    modes[modeIdx].amplitude = amplitudeFactor;
    modes[modeIdx].decay = 1.0f / decayFactor;  // Inverse - higher value = longer decay
    modes[modeIdx].phase = 0.0f;  // Start in phase
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
    // Default properties (Maple)
    params.density = 700.0f;          // kg/m³
    params.youngsModulus = 10.0e9f;   // 10 GPa (scaled down for simulation)
    params.poissonsRatio = 0.3f;      // Typical for wood
    params.dampingFactor = 0.02f;
    
    // Set material-specific parameters
    if (material == "Maple") {
        // Maple (medium-dense hardwood with balanced tone)
        params.density = 700.0f;
        params.youngsModulus = 10.0e9f;
        params.dampingFactor = 0.02f;
    }
    else if (material == "Birch") {
        // Birch (brighter sound with more attack)
        params.density = 650.0f;
        params.youngsModulus = 12.0e9f;
        params.dampingFactor = 0.015f;
    }
    else if (material == "Mahogany") {
        // Mahogany (warm and deep sound)
        params.density = 550.0f;
        params.youngsModulus = 8.0e9f;
        params.dampingFactor = 0.025f;
    }
    else if (material == "Metal") {
        // Metal (bright with long sustain) - using aluminum properties
        params.density = 2700.0f;
        params.youngsModulus = 70.0e9f;
        params.poissonsRatio = 0.33f;
        params.dampingFactor = 0.005f;
    }
    else if (material == "Acrylic") {
        // Acrylic (clear sound with less resonance)
        params.density = 1200.0f;
        params.youngsModulus = 3.0e9f;
        params.poissonsRatio = 0.4f;
        params.dampingFactor = 0.04f;
    }
    
    // Scale physical properties to work in our simulation
    // Real world values are too large for float precision in our simulation
    // This scaling preserves relative differences between materials
    params.youngsModulus *= 1.0e-9f;  // Scale gigapascals to simulation units
    params.density *= 0.001f;         // Scale kg/m³ to simulation units
    
    // Set the minimum and maximum frequency based on the material
    // These will be used as bounds for our physically calculated frequencies
    if (material == "Maple") {
        params.minFrequency = 60.0f;
        params.maxFrequency = 4000.0f;
    }
    else if (material == "Birch") {
        params.minFrequency = 80.0f;
        params.maxFrequency = 5000.0f;
    }
    else if (material == "Mahogany") {
        params.minFrequency = 50.0f;
        params.maxFrequency = 3500.0f;
    }
    else if (material == "Metal") {
        params.minFrequency = 100.0f;
        params.maxFrequency = 8000.0f;
    }
    else if (material == "Acrylic") {
        params.minFrequency = 70.0f;
        params.maxFrequency = 3000.0f;
    }
    
    // Calculate actual mode frequency ranges based on dimensions in initializeModesKernel
    printf("Material preset applied: %s\n", material.c_str());
    printf("Physical properties - Density: %.1f, Young's modulus: %.2f, Poisson's ratio: %.2f\n", 
           params.density * 1000.0f, params.youngsModulus * 1.0e9f, params.poissonsRatio);
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