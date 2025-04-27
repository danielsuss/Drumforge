#ifndef DRUMFORGE_BODY_KERNELS_CUH
#define DRUMFORGE_BODY_KERNELS_CUH

#include <cuda_runtime.h>
#include <vector_types.h>
#include <algorithm>
#include <string>

namespace drumforge {

/**
 * Structure representing a resonant mode in the body
 */
struct ResonantMode {
    float frequency;    // Resonant frequency in Hz
    float amplitude;    // Amplitude of this mode
    float decay;        // Decay time in seconds
    float phase;        // Phase offset (not currently used)
};

/**
 * Simplified structure containing parameters for body resonator kernels
 */
struct BodyKernelParams {
    // Basic body dimensions
    float radius;      // Body radius (same as membrane radius)
    float height;      // Body height relative to radius
    float thickness;   // Shell thickness relative to radius
    
    // Material properties (simplified)
    float density;        // Material density (relative)
    float stiffness;      // Material stiffness (simplified from Young's modulus)
    float damping;        // Global damping factor
    
    // Mode configuration
    int numModes;          // Total number of modes
    int circumferentialModes;  // Number of circumferential modes (n values)
    int axialModes;            // Number of axial/longitudinal modes (m values)
    
    // Constructor with defaults
    BodyKernelParams()
        : radius(5.0f), height(0.5f), thickness(0.01f),
          density(1.0f), stiffness(1.0f), damping(0.05f),
          numModes(32), circumferentialModes(4), axialModes(4) {}
};

// Core CUDA kernels
__global__ void initializeModesKernel(ResonantMode* modes, const BodyKernelParams params);
__global__ void resetBodyKernel(float* modeStates, float* modeVelocities, int numModes);
__global__ void updateBodyModesKernel(float* modeStates, float* modeVelocities,
                                   const ResonantMode* modes, const float* excitation,
                                   int numModes, float timestep);
__global__ void exciteModeKernel(float* excitation, int modeIndex, float amount, int numModes);
__global__ void exciteAllModesKernel(float* excitation, const float* inputExcitation, int numModes);

// Host-side wrapper functions
void initializeModes(ResonantMode* d_modes, const BodyKernelParams& params);
void resetBody(float* d_modeStates, float* d_modeVelocities, int numModes);
void updateBodyModes(float* d_modeStates, float* d_modeVelocities,
                   const ResonantMode* d_modes, const float* d_excitation,
                   int numModes, float timestep);
void exciteMode(float* d_excitation, int modeIndex, float amount, int numModes);
void exciteAllModes(float* d_excitation, const float* d_inputExcitation, int numModes);

// Helper functions
dim3 calculateOptimalBlockSize(int dataSize);
void setupMaterialPreset(BodyKernelParams& params, const std::string& material);
float calculateStableTimestep(const BodyKernelParams& params);

} // namespace drumforge

#endif // DRUMFORGE_BODY_KERNELS_CUH