#ifndef DRUMFORGE_COMPONENT_INTERFACE_H
#define DRUMFORGE_COMPONENT_INTERFACE_H

#include <string>
#include <memory>
#include <vector>  // Added for std::vector
#include <glm/glm.hpp>

namespace drumforge {

// Forward declarations
class SimulationManager;
class CudaMemoryManager;
class VisualizationManager;

// Structure to hold coupling data for interaction between components
struct CouplingData {
    // This will be expanded as we implement component interactions
    // Could include forces, pressures, displacements, etc.
};

/**
 * @brief Base interface for all simulation components
 * 
 * This abstract class defines the interface that all simulation components
 * must implement. Components are responsible for physics simulation of a 
 * specific part of the drum (membrane, air cavity, shell, etc.)
 */
class ComponentInterface {
public:
    // Virtual destructor for proper cleanup in derived classes
    virtual ~ComponentInterface() = default;
    
    // Initialize the component with the memory manager
    virtual void initialize() = 0;
    
    // Update the component state based on the current timestep
    virtual void update(float timestep) = 0;
    
    // Prepare component data for visualization (if necessary)
    virtual void prepareForVisualization() = 0;
    
    // Get coupling data to share with other components
    virtual CouplingData getInterfaceData() = 0;
    
    // Set coupling data received from other components
    virtual void setCouplingData(const CouplingData& data) = 0;
    
    // Get the component name for identification
    virtual std::string getName() const = 0;
    
    // Calculate a stable timestep for this component
    virtual float calculateStableTimestep() const = 0;
    
    // Get component's dimension requirements (2D, 3D)
    enum class DimensionRequirement {
        DIMENSION_2D,  // Component requires only X and Y dimensions (e.g., membrane)
        DIMENSION_3D   // Component requires X, Y, and Z dimensions (e.g., air cavity)
    };
    
    virtual DimensionRequirement getDimensionRequirement() const = 0;
    
    // Visualization-related methods
    
    /**
     * @brief Check if component supports visualization
     * 
     * @return true if the component can be rendered visually
     * @return false if the component is simulation-only
     */
    virtual bool isVisualizable() const { return false; }
    
    /**
     * @brief Initialize visualization resources for this component
     * 
     * This method is called by the VisualizationManager during setup.
     * Components should use this to initialize any OpenGL resources they need.
     * 
     * @param visManager Reference to the visualization manager
     */
    virtual void initializeVisualization(VisualizationManager& visManager) {}
    
    /**
     * @brief Provide component-specific visualization data
     * 
     * This method allows the component to provide visualization parameters to the
     * VisualizationManager, which handles the actual rendering.
     * 
     * @param visManager Reference to the visualization manager
     */
    virtual void visualize(VisualizationManager& visManager) {}

    // Audio-related methods
    virtual void initializeAudioChannels() {}
    virtual void updateAudio(float timestep) {}
    virtual bool hasAudio() const { return false; }

protected:
    std::vector<int> audioChannelIndices;  // Indices of channels registered with AudioManager
    bool audioChannelsInitialized = false;  // Flag to track if we've registered channels
};

} // namespace drumforge

#endif // DRUMFORGE_COMPONENT_INTERFACE_H