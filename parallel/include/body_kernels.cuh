#ifndef DRUMFORGE_BODY_KERNELS_CUH
#define DRUMFORGE_BODY_KERNELS_CUH

#include <cuda_runtime.h>
#include <vector_types.h>
#include <algorithm>
#include <string> // Add explicit include for string

namespace drumforge {

// Forward declaration to avoid circular include
struct ResonantMode;

/**
 * Structure containing parameters for body resonator kernels
 */
struct BodyKernelParams {
    // Basic body dimensions (matching the membrane's units)
    float radius;      // Body radius (same as membrane radius)
    float height;      // Body height relative to radius
    float thickness;   // Shell thickness relative to radius
    float cellSize;    // Grid cell size from simulation manager
    
    // Material properties
    float density;        // Material density (relative)
    float youngsModulus;  // Young's modulus (relative)
    float dampingFactor;  // Global damping factor
    
    // Modal synthesis settings
    int numModes;          // Number of resonant modes
    float minFrequency;    // Minimum resonant frequency (Hz)
    float maxFrequency;    // Maximum resonant frequency (Hz)
    float modeSpacing;     // Spacing between modes (1.0 = equal, >1.0 = stretched)
    
    // Additional physical parameters
    float poissonsRatio;    // Poisson's ratio
    
    // Mode-specific parameters
    int maxCircumferentialModes;  // Max number of circumferential modes (n)
    int maxAxialModes;            // Max number of axial/longitudinal modes (m)
    
    // Constructor update
    BodyKernelParams()
        : radius(5.0f), height(10.0f), thickness(0.5f), cellSize(1.0f),
          density(0.7f), youngsModulus(1.0f), dampingFactor(0.02f),
          poissonsRatio(0.3f), // Default Poisson's ratio for wood
          numModes(48), minFrequency(60.0f), maxFrequency(4000.0f), modeSpacing(1.1f),
          maxCircumferentialModes(8), maxAxialModes(6) {}
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
// Mark this as host-only since it uses std::string
__host__ void setupMaterialPreset(BodyKernelParams& params, const std::string& material);
float calculateStableTimestep(const BodyKernelParams& params);

} // namespace drumforge

#endif // DRUMFORGE_BODY_KERNELS_CUH