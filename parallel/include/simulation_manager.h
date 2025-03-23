#ifndef DRUMFORGE_SIMULATION_MANAGER_H
#define DRUMFORGE_SIMULATION_MANAGER_H

#include "component_interface.h"
#include "cuda_memory_manager.h"
#include <vector>
#include <memory>
#include <string>
#include <unordered_map>

namespace drumforge {

/**
 * @brief Global simulation parameters
 * 
 * This structure holds parameters that affect the entire simulation,
 * including grid specifications shared across all components
 */
struct SimulationParameters {
    // Simulation control parameters
    float timeScale = 1.0f;         // Simulation speed multiplier
    bool pauseSimulation = false;   // Pause flag
    float globalDamping = 0.001f;   // Global damping coefficient
    
    // Grid specifications (shared by all components)
    int gridSizeX = 64;             // Grid size in X dimension
    int gridSizeY = 64;             // Grid size in Y dimension 
    int gridSizeZ = 32;             // Grid size in Z dimension
    float cellSize = 1.0f;          // Physical size of each grid cell in world units
    
    // Material properties
    float airDensity = 1.2f;        // Air density (kg/m^3)
    float speedOfSound = 343.0f;    // Speed of sound in air (m/s)
};

/**
 * @brief Manager for coordinating all simulation components
 * 
 * This class orchestrates the simulation of all components,
 * handles their interactions, manages the simulation loop,
 * and maintains shared parameters like grid specifications.
 */
class SimulationManager {
private:
    // Singleton instance
    static SimulationManager* instance;
    
    // Components in the simulation
    std::vector<std::shared_ptr<ComponentInterface>> components;
    
    // Mapping between component IDs for coupling
    struct CouplingPair {
        std::shared_ptr<ComponentInterface> source;
        std::shared_ptr<ComponentInterface> target;
    };
    std::vector<CouplingPair> couplings;
    
    // Simulation state
    float currentTime = 0.0f;
    float accumulatedTime = 0.0f;
    SimulationParameters params;
    
    // Reference to memory manager
    CudaMemoryManager& memoryManager;
    
    // Private constructor for singleton
    SimulationManager();
    
public:
    // No copying or assignment
    SimulationManager(const SimulationManager&) = delete;
    SimulationManager& operator=(const SimulationManager&) = delete;
    
    // Get singleton instance
    static SimulationManager& getInstance();
    
    // Initialize the simulation
    void initialize();
    
    // Shut down the simulation
    void shutdown();
    
    // Add a component to the simulation
    void addComponent(std::shared_ptr<ComponentInterface> component);
    
    // Set up coupling between two components
    void setupCoupling(std::shared_ptr<ComponentInterface> source, 
                      std::shared_ptr<ComponentInterface> target);
    
    // Advance simulation by the specified time step
    void advance(float deltaTime);
    
    // Update simulation parameters
    void updateParameters(const SimulationParameters& newParams);
    
    // Get current simulation parameters (components use this to access grid specs)
    const SimulationParameters& getParameters() const;
    
    // Get component by name (useful for specific interactions)
    std::shared_ptr<ComponentInterface> getComponentByName(const std::string& name);
    
    // Calculate a stable timestep based on all components
    float calculateStableTimestep() const;
    
    // Set grid specifications (should be called before components are initialized)
    void setGridSpecifications(int sizeX, int sizeY, int sizeZ, float cellSize);
};

} // namespace drumforge

#endif // DRUMFORGE_SIMULATION_MANAGER_H