#ifndef DRUMFORGE_BODY_KERNELS_CUH
#define DRUMFORGE_BODY_KERNELS_CUH

#include <cuda_runtime.h>
#include <vector_types.h>
#include <glm/glm.hpp>
#include <string>
#include <algorithm>

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
    
    // Excitation settings
    float excitationDecay; // How quickly excitation decays
    
    // Constructor with default values
    BodyKernelParams()
        : radius(5.0f), height(10.0f), thickness(0.5f), cellSize(1.0f),
          density(0.7f), youngsModulus(1.0f), dampingFactor(0.02f),
          numModes(64), minFrequency(60.0f), maxFrequency(4000.0f), modeSpacing(1.1f),
          excitationDecay(0.1f) {}
};

/**
 * Initialize the resonant modes based on material parameters
 * 
 * @param modes Output array of ResonantMode structures
 * @param params Kernel parameters
 */
__global__ void initializeModesKernel(ResonantMode* modes, const BodyKernelParams params);

/**
 * Reset the body to its rest state
 * 
 * @param modeStates Current state of each mode (displacement)
 * @param modeVelocities Velocities of each mode
 * @param numModes Number of modes
 */
__global__ void resetBodyKernel(float* modeStates, float* modeVelocities, int numModes);

/**
 * Update the modal synthesis simulation
 * 
 * @param modeStates Current state of each mode (displacement)
 * @param modeVelocities Velocities of each mode
 * @param modes Array of mode parameters
 * @param excitation Array of excitation values for each mode
 * @param numModes Number of modes
 * @param timestep Simulation time step
 */
__global__ void updateBodyModesKernel(float* modeStates, float* modeVelocities,
                                    const ResonantMode* modes, const float* excitation,
                                    int numModes, float timestep);

/**
 * Apply an excitation to a specific mode
 * 
 * @param excitation Output excitation array
 * @param modeIndex Mode to excite (-1 for all modes)
 * @param amount Excitation amount
 * @param numModes Number of modes
 */
__global__ void exciteModeKernel(float* excitation, int modeIndex, float amount, int numModes);

/**
 * Apply a pattern of excitation across all modes (e.g., from membrane impact)
 * 
 * @param excitation Output excitation array
 * @param inputExcitation Input excitation pattern
 * @param numModes Number of modes
 */
__global__ void exciteAllModesKernel(float* excitation, const float* inputExcitation,
                                   int numModes);

/**
 * Calculate output sample from modal states for a given microphone position
 * 
 * @param modeStates Current state of each mode (displacement)
 * @param modes Array of mode parameters
 * @param micPositionX Microphone X position (normalized 0-1)
 * @param micPositionY Microphone Y position (normalized 0-1)
 * @param micHeight Microphone height position (normalized 0-1)
 * @param numModes Number of modes
 * @return The computed sample value
 */
__global__ void sampleBodyAtPositionKernel(float* output, const float* modeStates, 
                                         const ResonantMode* modes,
                                         float micPositionX, float micPositionY, 
                                         float micHeight, int numModes);

// Host-side function to launch the initialization kernel
void initializeModes(ResonantMode* d_modes, const BodyKernelParams& params);

// Host-side function to launch the reset kernel
void resetBody(float* d_modeStates, float* d_modeVelocities, int numModes);

// Host-side function to launch the update kernel
void updateBodyModes(float* d_modeStates, float* d_modeVelocities,
                   const ResonantMode* d_modes, const float* d_excitation,
                   int numModes, float timestep);

// Host-side function to launch the excite mode kernel
void exciteMode(float* d_excitation, int modeIndex, float amount, int numModes);

// Host-side function to launch the excite all modes kernel
void exciteAllModes(float* d_excitation, const float* d_inputExcitation, int numModes);

// Host-side function to sample body at microphone position
float sampleBodyAtPosition(float* d_output, const float* d_modeStates, const ResonantMode* d_modes,
                         float micPositionX, float micPositionY, float micHeight, int numModes);

// Helper to calculate optimal CUDA thread and block configuration
dim3 calculateOptimalBlockSize(int dataSize);

// Helper to set up material preset parameters
void setupMaterialPreset(BodyKernelParams& params, const std::string& material);

// Helper to calculate a stable timestep for modal synthesis
float calculateStableTimestep(const BodyKernelParams& params);

} // namespace drumforge

#endif // DRUMFORGE_BODY_KERNELS_CUH