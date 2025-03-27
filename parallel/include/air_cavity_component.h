#ifndef DRUMFORGE_AIR_CAVITY_COMPONENT_H
#define DRUMFORGE_AIR_CAVITY_COMPONENT_H

#include "component_interface.h"
#include "cuda_memory_manager.h"
#include "cuda_gl_buffer.h"
#include "simulation_manager.h"
#include <memory>
#include <vector>
#include <string>
#include <glm/glm.hpp>

namespace drumforge {

// Forward declarations
struct AirCavityKernelParams;
class VisualizationManager;

/**
 * Structure representing a virtual microphone
 */
struct VirtualMicrophone {
    glm::vec3 position;   // Position in world coordinates
    int gridX, gridY, gridZ; // Position in grid coordinates
    float gain;           // Microphone gain
    std::vector<float> samples; // Recorded samples buffer
    int sampleRate;       // Sample rate in Hz
};

/**
 * AirCavityComponent simulates acoustic wave propagation in a 3D volume 
 * using CUDA-accelerated finite difference time domain (FDTD) methods 
 * to solve the 3D acoustic wave equation.
 */
class AirCavityComponent : 
    public ComponentInterface,
    public std::enable_shared_from_this<AirCavityComponent> {
private:
    // Reference to memory manager and simulation manager
    CudaMemoryManager& memoryManager;
    SimulationManager& simulationManager;
    
    // Physical parameters
    float speedOfSound;     // Speed of sound in air (m/s)
    float density;          // Density of air (kg/m^3)
    float dampingCoefficient; // Damping coefficient for stability and absorption

    // Grid parameters (set during initialization from SimulationManager)
    int cavityWidth;        // Width of the 3D grid (X dimension)
    int cavityHeight;       // Height of the 3D grid (Y dimension)
    int cavityDepth;        // Depth of the 3D grid (Z dimension)
    float cellSize;         // Physical size of each grid cell (m)

    // Component positioning
    GridRegion region;      // The region allocated from the global grid

    // CUDA device buffers (obtained from CudaMemoryManager)
    std::shared_ptr<CudaBuffer<float>> d_pressure;      // Current pressure field
    std::shared_ptr<CudaBuffer<float>> d_prevPressure;  // Previous pressure field
    std::shared_ptr<CudaBuffer<float>> d_velocityX;     // X-component of velocity (staggered grid)
    std::shared_ptr<CudaBuffer<float>> d_velocityY;     // Y-component of velocity (staggered grid)
    std::shared_ptr<CudaBuffer<float>> d_velocityZ;     // Z-component of velocity (staggered grid)

    // OpenGL interop buffer for visualization (will be null if interop is disabled)
    std::shared_ptr<CudaGLBuffer> glInteropBuffer;

    // OpenGL resource IDs
    unsigned int vaoId;        // Vertex Array Object ID
    unsigned int eboId;        // Element Buffer Object ID
    
    // Host-side data for CPU access
    mutable std::vector<float> h_pressure;  // Cached pressure field for visualization & audio
    int currentSliceZ;         // Current Z slice for 2D visualization

    // Component name
    std::string name;

    // Kernel parameters wrapper (to avoid including CUDA headers here)
    std::unique_ptr<AirCavityKernelParams> kernelParams;

    // Virtual microphones for audio output
    std::vector<VirtualMicrophone> microphones;

    // Visualization-related members
    enum class VisualizationMode {
        SLICE_XY,     // Show a 2D slice in the XY plane
        SLICE_XZ,     // Show a 2D slice in the XZ plane
        SLICE_YZ,     // Show a 2D slice in the YZ plane
        VOLUME        // Show 3D volume (more advanced)
    };
    VisualizationMode visualizationMode;

    // Audio output parameters
    int sampleRate;           // Audio sample rate in Hz
    bool audioEnabled;        // Whether audio output is enabled
    std::vector<float> audioBuffer; // Buffer for audio output samples

    // Private utility methods
    void applyBoundaryConditions();
    void updateVisualizationData();
    void updateMicrophoneSamples(float timestep);

public:
    // Constructor
    AirCavityComponent(
        const std::string& name,
        float speedOfSound = 343.0f,
        float density = 1.2f,
        float dampingCoefficient = 0.001f
    );
    
    // Deleted copy constructor and assignment operator
    AirCavityComponent(const AirCavityComponent&) = delete;
    AirCavityComponent& operator=(const AirCavityComponent&) = delete;

    // Destructor
    ~AirCavityComponent();

    //--------------------------------------------------------------------------
    // ComponentInterface methods
    //--------------------------------------------------------------------------
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

    //--------------------------------------------------------------------------
    // AirCavity-specific methods
    //--------------------------------------------------------------------------
    
    // Simulation methods
    void reset();  // Reset the simulation to initial state
    
    // Add a pressure impulse at a specific position
    void addPressureImpulse(float x, float y, float z, float strength, float radius);
    
    // Virtual microphone management
    int addMicrophone(float x, float y, float z, int sampleRate = 44100, float gain = 1.0f);
    void removeMicrophone(int index);
    const std::vector<float>& getMicrophoneSamples(int index) const;
    
    // Audio playback control
    void enableAudio(bool enable);
    bool isAudioEnabled() const { return audioEnabled; }
    void setSampleRate(int rate);
    int getSampleRate() const { return sampleRate; }
    
    // Parameter setters - for GUI control
    void setSpeedOfSound(float speed);
    void setDensity(float density);
    void setDampingCoefficient(float damping);
    
    // Visualization control
    void setVisualizationMode(VisualizationMode mode);
    void setCurrentSliceZ(int z); // For 2D slice visualization
    VisualizationMode getVisualizationMode() const { return visualizationMode; }
    
    // Pressure field access
    float getPressureAt(float x, float y, float z) const;
    float getPressureAtGrid(int x, int y, int z) const;
    const std::vector<float>& getPressureField() const;
    
    // Grid information
    int getCavityWidth() const { return cavityWidth; }
    int getCavityHeight() const { return cavityHeight; }
    int getCavityDepth() const { return cavityDepth; }
    float getCellSize() const { return cellSize; }
    
    // Physical parameters access
    float getSpeedOfSound() const { return speedOfSound; }
    float getDensity() const { return density; }
    float getDampingCoefficient() const { return dampingCoefficient; }
    
    // Coordinate conversion utilities
    glm::vec3 gridToWorld(int x, int y, int z) const;
    void worldToGrid(float x, float y, float z, int& outX, int& outY, int& outZ) const;
    
    // OpenGL resource access for visualization manager
    unsigned int getVAO() const { return vaoId; }
    unsigned int getEBO() const { return eboId; }
    int getVertexCount() const;
    int getIndexCount() const;
    int getCurrentSliceZ() const { return currentSliceZ; }
};

} // namespace drumforge

#endif // DRUMFORGE_AIR_CAVITY_COMPONENT_H