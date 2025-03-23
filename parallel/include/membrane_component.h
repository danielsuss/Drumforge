#ifndef DRUMFORGE_MEMBRANE_COMPONENT_H
#define DRUMFORGE_MEMBRANE_COMPONENT_H

#include "component_interface.h"
#include "cuda_memory_manager.h"
#include "cuda_gl_buffer.h"
#include "simulation_manager.h"
#include <memory>
#include <vector>
#include <string>
#include <glm/glm.hpp>

namespace drumforge {

// Forward declaration of CUDA kernel parameters
struct MembraneKernelParams;

/**
 * MembraneComponent simulates a circular drum membrane using CUDA-accelerated
 * finite difference time domain (FDTD) methods to solve the 2D wave equation.
 */
class MembraneComponent : public ComponentInterface {
private:
    // Reference to memory manager and simulation manager
    CudaMemoryManager& memoryManager;
    SimulationManager& simulationManager;
    
    // Physical parameters
    float tension;       // Membrane tension (affects wave speed)
    float damping;       // Damping coefficient (affects decay rate)
    float radius;        // Radius of the circular membrane

    // Grid parameters (set during initialization from SimulationManager)
    int gridSizeX;       // Width of the component's grid
    int gridSizeY;       // Height of the component's grid
    float cellSize;      // Physical size of each grid cell

    // Component positioning
    GridRegion region;   // The region allocated from the global grid

    // CUDA device buffers (obtained from CudaMemoryManager)
    std::shared_ptr<CudaBuffer<float>> d_heights;      // Current heights
    std::shared_ptr<CudaBuffer<float>> d_prevHeights;  // Previous heights
    std::shared_ptr<CudaBuffer<float>> d_velocities;   // Velocities
    std::shared_ptr<CudaBuffer<int>> d_circleMask;     // Circular mask (1 inside, 0 outside)

    // OpenGL interop buffer for visualization
    std::shared_ptr<CudaGLBuffer> glInteropBuffer;  // Obtained from CudaMemoryManager

    // Host-side data for CPU access (mostly for I/O)
    std::vector<float> h_heights;

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

    // Membrane-specific methods
    void applyImpulse(float x, float y, float strength);
    void reset();

    // Accessors
    const std::vector<float>& getHeights() const;
    float getHeight(int x, int y) const;
    int getGridSizeX() const { return gridSizeX; }
    int getGridSizeY() const { return gridSizeY; }
    float getRadius() const { return radius; }
    
    // Check if a point is inside the circular membrane
    bool isInsideCircle(int x, int y) const;
};

} // namespace drumforge

#endif // DRUMFORGE_MEMBRANE_COMPONENT_H