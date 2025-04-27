#ifndef DRUMFORGE_BODY_COMPONENT_H
#define DRUMFORGE_BODY_COMPONENT_H

#include "component_interface.h"
#include "cuda_memory_manager.h"
#include "simulation_manager.h"
#include "audio_manager.h"
#include <memory>
#include <vector>
#include <string>
#include <glm/glm.hpp>

namespace drumforge {

// Forward declarations
struct BodyKernelParams;
struct ResonantMode;
class VisualizationManager;

/**
 * Structure representing a virtual microphone on the body
 */
struct BodyVirtualMicrophone {
    glm::vec2 position;   // Normalized position [0,1] on the body surface
    float height;         // Normalized height position [0,1]
    float gain;           // Microphone gain factor
    bool enabled;         // Whether the microphone is active
    std::string name;     // Optional name for the microphone
};

/**
 * BodyComponent simulates a drum shell using modal synthesis.
 * It models the resonant properties of the shell material using
 * a bank of resonators processed on the GPU.
 */
class BodyComponent : 
    public ComponentInterface,
    public std::enable_shared_from_this<BodyComponent> {
private:
    // Reference to memory manager and simulation manager
    CudaMemoryManager& memoryManager;
    SimulationManager& simulationManager;
    
    // Physical parameters
    float radius;          // Body radius (same as membrane radius)
    float height;          // Body height
    float thickness;       // Shell thickness
    float masterGain;      // Master output gain
    bool useMixedOutput;   // Whether to mix all microphones or use them separately
    std::string material;  // Shell material name
    
    // Grid parameters
    float cellSize;        // Physical size of each grid cell
    
    // Component positioning
    GridRegion region;     // The region allocated from the global grid
    
    // CUDA device buffers (obtained from CudaMemoryManager)
    std::shared_ptr<CudaBuffer<float>> d_modeStates;      // Current states
    std::shared_ptr<CudaBuffer<float>> d_modeVelocities;  // Velocities
    std::shared_ptr<CudaBuffer<ResonantMode>> d_modes;    // Mode parameters
    std::shared_ptr<CudaBuffer<float>> d_excitation;      // Input excitation buffer
    
    // Host-side data for CPU access (mostly for I/O)
    mutable std::vector<float> h_modeStates;
    mutable std::vector<float> h_modeVelocities;
    mutable std::vector<ResonantMode> h_modes;
    
    // Microphones for sampling the body
    std::vector<BodyVirtualMicrophone> microphones;
    
    // Component name
    std::string name;
    
    // Kernel parameters wrapper
    std::unique_ptr<BodyKernelParams> kernelParams;
    
    // OpenGL resource IDs for visualization
    unsigned int vaoId;        // Vertex Array Object ID
    unsigned int eboId;        // Element Buffer Object ID
    
    // Private utility methods
    void setupDefaultModes();
    void setupMaterialPreset(const std::string& material);
    float sampleMicrophonePosition(const BodyVirtualMicrophone& mic);
    int getNumWireframeIndices() const;

public:
    // Constructor
    BodyComponent(
        const std::string& name,
        float radius,            // Should match membrane radius
        float height = 0.4f,     // Default height as 40% of radius
        float thickness = 0.01f, // Default thickness as 1% of radius
        const std::string& material = "Maple"
    );
    
    // Deleted copy constructor and assignment operator
    BodyComponent(const BodyComponent&) = delete;
    BodyComponent& operator=(const BodyComponent&) = delete;
    
    // Destructor
    ~BodyComponent();
    
    // ComponentInterface methods
    void initialize() override;
    void update(float timestep) override;
    void prepareForVisualization() override;
    CouplingData getInterfaceData() override;
    void setCouplingData(const CouplingData& data) override;
    std::string getName() const override;
    float calculateStableTimestep() const override;
    DimensionRequirement getDimensionRequirement() const override {
        return DimensionRequirement::DIMENSION_3D;
    }
    
    // Visualization-related methods
    bool isVisualizable() const override { return true; }
    void initializeVisualization(VisualizationManager& visManager) override;
    void visualize(VisualizationManager& visManager) override;
    
    // Audio-related methods
    void initializeAudioChannels() override;
    void updateAudio(float timestep) override;
    bool hasAudio() const override { return true; }
    
    // Body-specific methods
    void reset();
    void exciteMode(int modeIndex, float amount);
    void exciteAllModes(const std::vector<float>& excitation);
    
    // Parameter setters
    void setMaterial(const std::string& newMaterial);
    void setHeight(float newHeight);
    void setThickness(float newThickness);
    
    // Microphone management
    int addMicrophone(float x, float y, float height, float gain = 1.0f, const std::string& name = "");
    void removeMicrophone(int index);
    void clearAllMicrophones();
    int getMicrophoneCount() const { return static_cast<int>(microphones.size()); }
    const BodyVirtualMicrophone& getMicrophone(int index) const;
    void setMicrophonePosition(int index, float x, float y, float height);
    void setMicrophoneGain(int index, float gain);
    void enableMicrophone(int index, bool enabled);
    void setMasterGain(float gain) { masterGain = gain; }
    float getMasterGain() const { return masterGain; }
    void setUseMixedOutput(bool useMixed) { useMixedOutput = useMixed; }
    bool getUseMixedOutput() const { return useMixedOutput; }
    
    // Microphone presets
    void setupDefaultMicrophones();
    
    // Debug functions
    void reportModeFrequencies();
    
    // Accessors
    float getRadius() const { return radius; }
    float getHeight() const { return height; }
    float getThickness() const { return thickness; }
    const std::string& getMaterial() const { return material; }
    const std::vector<float>& getModeStates() const;
    unsigned int getVAO() const { return vaoId; }
    unsigned int getEBO() const { return eboId; }
};

} // namespace drumforge

#endif // DRUMFORGE_BODY_COMPONENT_H