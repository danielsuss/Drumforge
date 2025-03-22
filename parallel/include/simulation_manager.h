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
 * This structure holds parameters that affect the entire simulation
 */
struct SimulationParameters {
    float timeScale = 1.0f;         // Simulation speed multiplier
    bool pauseSimulation = false;   // Pause flag
    float globalDamping = 0.001f;   // Global damping coefficient
    // Add more global parameters as needed
};

/**
 * @brief Manager for coordinating all simulation components
 * 
 * This class orchestrates the simulation of all components,
 * handles their interactions, and manages the simulation loop.
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
    
    // Get current simulation parameters
    const SimulationParameters& getParameters() const;
    
    // Get component by name (useful for specific interactions)
    std::shared_ptr<ComponentInterface> getComponentByName(const std::string& name);
    
    // Calculate a stable timestep based on all components
    float calculateStableTimestep() const;
};

} // namespace drumforge

#endif // DRUMFORGE_SIMULATION_MANAGER_H