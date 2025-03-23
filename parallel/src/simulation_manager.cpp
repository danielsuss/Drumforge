#include "simulation_manager.h"
#include <iostream>
#include <algorithm>
#include <limits>

namespace drumforge {

// Initialize static instance pointer
SimulationManager* SimulationManager::instance = nullptr;

// Constructor
SimulationManager::SimulationManager()
    : memoryManager(CudaMemoryManager::getInstance()) {
    std::cout << "SimulationManager created" << std::endl;
}

// Get singleton instance
SimulationManager& SimulationManager::getInstance() {
    if (instance == nullptr) {
        instance = new SimulationManager();
    }
    return *instance;
}

// Initialize the simulation
void SimulationManager::initialize() {
    std::cout << "Initializing simulation..." << std::endl;
    
    // Initialize CUDA memory manager
    memoryManager.initialize();
    
    // Initialize all components
    for (auto& component : components) {
        component->initialize();
    }
    
    std::cout << "Simulation initialized with " << components.size() << " components" << std::endl;
    std::cout << "Grid dimensions: " << params.gridSizeX << "x" << params.gridSizeY << "x" 
              << params.gridSizeZ << " (cell size: " << params.cellSize << ")" << std::endl;
}

// Shut down the simulation
void SimulationManager::shutdown() {
    std::cout << "Shutting down simulation..." << std::endl;
    
    // Clear all components
    components.clear();
    couplings.clear();
    
    // Reset singleton instance
    delete instance;
    instance = nullptr;
}

// Add a component to the simulation
void SimulationManager::addComponent(std::shared_ptr<ComponentInterface> component) {
    components.push_back(component);
    std::cout << "Added component: " << component->getName() << std::endl;
}

// Set up coupling between two components
void SimulationManager::setupCoupling(std::shared_ptr<ComponentInterface> source, 
                                     std::shared_ptr<ComponentInterface> target) {
    couplings.push_back({source, target});
    std::cout << "Set up coupling from " << source->getName() 
              << " to " << target->getName() << std::endl;
}

// Advance simulation by the specified time step
void SimulationManager::advance(float deltaTime) {
    if (params.pauseSimulation) {
        return;
    }
    
    // Apply time scaling
    float scaledDeltaTime = deltaTime * params.timeScale;
    
    // Accumulate time for fixed-step updates
    accumulatedTime += scaledDeltaTime;
    
    // Calculate a stable timestep
    float stableTimestep = calculateStableTimestep();
    
    // Update components with fixed time steps
    while (accumulatedTime >= stableTimestep) {
        // First update all components
        for (auto& component : components) {
            component->update(stableTimestep);
        }
        
        // Then process all couplings
        for (auto& coupling : couplings) {
            // Exchange data between coupled components
            CouplingData data = coupling.source->getInterfaceData();
            coupling.target->setCouplingData(data);
        }
        
        // Update simulation time
        currentTime += stableTimestep;
        accumulatedTime -= stableTimestep;
    }
}

// Update simulation parameters
void SimulationManager::updateParameters(const SimulationParameters& newParams) {
    params = newParams;
}

// Get current simulation parameters
const SimulationParameters& SimulationManager::getParameters() const {
    return params;
}

// Get component by name
std::shared_ptr<ComponentInterface> SimulationManager::getComponentByName(const std::string& name) {
    for (auto& component : components) {
        if (component->getName() == name) {
            return component;
        }
    }
    return nullptr;
}

// Calculate a stable timestep based on all components
float SimulationManager::calculateStableTimestep() const {
    float minTimestep = std::numeric_limits<float>::max();
    
    // Find the smallest stable timestep among all components
    for (const auto& component : components) {
        float componentTimestep = component->calculateStableTimestep();
        minTimestep = std::min(minTimestep, componentTimestep);
    }
    
    // If no components or invalid timestep, return a default
    if (minTimestep == std::numeric_limits<float>::max()) {
        return 0.001f; // Default timestep of 1ms
    }
    
    return minTimestep;
}

// Set grid specifications
void SimulationManager::setGridSpecifications(int sizeX, int sizeY, int sizeZ, float cellSize) {
    // Store the new grid parameters
    params.gridSizeX = sizeX;
    params.gridSizeY = sizeY;
    params.gridSizeZ = sizeZ;
    params.cellSize = cellSize;
    
    std::cout << "Grid specifications updated: " << sizeX << "x" << sizeY << "x" 
              << sizeZ << " (cell size: " << cellSize << ")" << std::endl;
              
    // Note: This should be called before components are initialized
    // as components will use these specifications during initialization
}

} // namespace drumforge