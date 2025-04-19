#ifndef DRUMFORGE_MEMBRANE_COMPONENT_H
#define DRUMFORGE_MEMBRANE_COMPONENT_H

#include "component_interface.h"
#include "cuda_memory_manager.h"
#include "cuda_gl_buffer.h"
#include "simulation_manager.h"
#include "audio_manager.h"
#include <memory>
#include <vector>
#include <string>
#include <glm/glm.hpp>

namespace drumforge {

// Forward declarations
struct MembraneKernelParams;
class VisualizationManager;

/**
 * Structure representing a virtual microphone on the membrane
 */
struct MembraneVirtualMicrophone {
    glm::vec2 position;   // Normalized position [0,1] on the membrane
    float gain;           // Microphone gain factor
    bool enabled;         // Whether the microphone is active
    std::string name;     // Optional name for the microphone (useful for UI)
};

/**
 * MembraneComponent simulates a circular drum membrane using CUDA-accelerated
 * finite difference time domain (FDTD) methods to solve the 2D wave equation.
 */
class MembraneComponent : 
    public ComponentInterface,
    public std::enable_shared_from_this<MembraneComponent> {
private:
    // Reference to memory manager and simulation manager
    CudaMemoryManager& memoryManager;
    SimulationManager& simulationManager;
    
    // Physical parameters
    float tension;       // Membrane tension (affects wave speed)
    float damping;       // Damping coefficient (affects decay rate)
    float radius;        // Radius of the circular membrane

    // Grid parameters (set during initialization from SimulationManager)
    int membraneWidth;    // Width of the component's data array
    int membraneHeight;   // Height of the component's data array
    float cellSize;       // Physical size of each grid cell

    // Component positioning
    GridRegion region;   // The region allocated from the global grid

    // CUDA device buffers (obtained from CudaMemoryManager)
    std::shared_ptr<CudaBuffer<float>> d_heights;      // Current heights
    std::shared_ptr<CudaBuffer<float>> d_prevHeights;  // Previous heights
    std::shared_ptr<CudaBuffer<float>> d_velocities;   // Velocities
    std::shared_ptr<CudaBuffer<int>> d_circleMask;     // Circular mask (1 inside, 0 outside)

    // OpenGL interop buffer for visualization
    std::shared_ptr<CudaGLBuffer> glInteropBuffer;  // Obtained from CudaMemoryManager

    // OpenGL resource IDs
    unsigned int vaoId;        // Vertex Array Object ID
    unsigned int eboId;        // Element Buffer Object ID
    
    // Host-side data for CPU access (mostly for I/O)
    mutable std::vector<float> h_heights;

    // Component name
    std::string name;

    // Kernel parameters wrapper (to avoid including CUDA headers here)
    std::unique_ptr<MembraneKernelParams> kernelParams;

    // Impulse parameters for animation/debugging
    struct ImpulseParams {
        float x, y;     // Position
        float strength;  // Amplitude
        bool active;     // Whether to apply this impulse
    };
    ImpulseParams pendingImpulse;

    // Audio sampling point (legacy - replaced by microphones)
    glm::vec2 audioSamplePoint;
    float audioGain;

    // Virtual microphones for audio output
    std::vector<MembraneVirtualMicrophone> microphones;
    float masterGain;         // Master gain applied to the final mix
    bool useMixedOutput;      // Whether to mix all microphones or use them separately

    // Private utility methods
    void updateBoundaryConditions();
    float calculateWaveSpeed() const;

public:
    // Constructor
    MembraneComponent(
        const std::string& name,
        float radius,
        float tension = 1.0f,
        float damping = 0.01f
    );
    
    // Deleted copy constructor and assignment operator
    MembraneComponent(const MembraneComponent&) = delete;
    MembraneComponent& operator=(const MembraneComponent&) = delete;

    // Destructor
    ~MembraneComponent();

    // ComponentInterface methods
    void initialize() override;
    void update(float timestep) override;
    void prepareForVisualization() override;
    CouplingData getInterfaceData() override;
    void setCouplingData(const CouplingData& data) override;
    std::string getName() const override;
    float calculateStableTimestep() const override;
    DimensionRequirement getDimensionRequirement() const override {
        return DimensionRequirement::DIMENSION_2D;
    }
    
    // Visualization-related methods
    bool isVisualizable() const override { return true; }
    void initializeVisualization(VisualizationManager& visManager) override;
    void visualize(VisualizationManager& visManager) override;

    // Membrane-specific methods
    void applyImpulse(float x, float y, float strength);
    void reset();

    // Parameter setters - for GUI control
    void setRadius(float newRadius);
    void setTension(float newTension);
    void setDamping(float newDamping);

    // Accessors
    const std::vector<float>& getHeights() const;
    float getHeight(int x, int y) const;
    int getMembraneWidth() const { return membraneWidth; }
    int getMembraneHeight() const { return membraneHeight; }
    float getRadius() const { return radius; }
    float getTension() const { return tension; }
    float getDamping() const { return damping; }
    
    // OpenGL resource access for visualization manager
    unsigned int getVAO() const { return vaoId; }
    unsigned int getEBO() const { return eboId; }
    int getVertexCount() const { return membraneWidth * membraneHeight; }
    int getIndexCount() const;
    
    // Check if a point is inside the circular membrane
    bool isInsideCircle(int x, int y) const;

    // Legacy audio methods (deprecated)
    void setAudioSamplePoint(float x, float y);
    void setAudioGain(float gain);
    // void updateAudio(float timestep);

    void initializeAudioChannels() override;
    void updateAudio(float timestep) override;
    bool hasAudio() const override { return true; }
    
    // Microphone management
    int addMicrophone(float x, float y, float gain = 1.0f, const std::string& name = "");
    void removeMicrophone(int index);
    void clearAllMicrophones();
    int getMicrophoneCount() const { return static_cast<int>(microphones.size()); }
    const MembraneVirtualMicrophone& getMicrophone(int index) const;
    void setMicrophonePosition(int index, float x, float y);
    void setMicrophoneGain(int index, float gain);
    void enableMicrophone(int index, bool enabled);
    void setMasterGain(float gain) { masterGain = gain; }
    float getMasterGain() const { return masterGain; }
    void setUseMixedOutput(bool useMixed) { useMixedOutput = useMixed; }
    bool getUseMixedOutput() const { return useMixedOutput; }
    
    // Microphone configuration presets
    void setupSingleCenterMicrophone();
    void setupStereoMicrophones();
    void setupQuadMicrophones();
    void setupCircularMicrophones(int count, float radius = 0.4f);
};

} // namespace drumforge

#endif // DRUMFORGE_MEMBRANE_COMPONENT_H