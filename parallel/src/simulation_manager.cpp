#include "simulation_manager.h"
#include <iostream>
#include <algorithm>
#include <limits>
#include <stdexcept>
#include "audio_manager.h"

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
    allocatedRegions.clear();
    
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
            
            // Only print if there's an impact
            if (data.hasImpact) {
                std::cout << "Coupling data from " << coupling.source->getName() 
                        << " to " << coupling.target->getName() 
                        << ": strength=" << data.impactStrength
                        << ", position=(" << data.impactPosition.x 
                        << "," << data.impactPosition.y << ")" << std::endl;
            }
            
            coupling.target->setCouplingData(data);
        }
        
        // Update audio channel values for all components (but don't process audio yet)
        for (auto& component : components) {
            if (component->hasAudio()) {
                component->updateAudio(stableTimestep);
            }
        }
        
        // Process audio once per simulation step - either mixed or individual channels
        AudioManager& audioManager = AudioManager::getInstance();
        
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
    
    minTimestep = 0.05f; // Minimum timestep for simulation

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

//--------------------------------------------------------------------------
// Grid coordination functions implementation
//--------------------------------------------------------------------------

GridRegion SimulationManager::allocateGridRegion(
    std::shared_ptr<ComponentInterface> component,
    int startX, int startY, int startZ,
    int sizeX, int sizeY, int sizeZ) {
    
    // Validate input
    if (startX < 0 || startY < 0 || startZ < 0 ||
        startX + sizeX > params.gridSizeX ||
        startY + sizeY > params.gridSizeY ||
        startZ + sizeZ > params.gridSizeZ) {
        std::cerr << "Error: Requested grid region is outside global grid bounds" << std::endl;
        return GridRegion(); // Return empty region on error
    }
    
    // Check for overlap with existing regions
    for (const auto& region : allocatedRegions) {
        // Simple check: if this new region's bounding box overlaps with an existing region
        bool overlaps = !(
            startX + sizeX <= region.startX ||
            startY + sizeY <= region.startY ||
            startZ + sizeZ <= region.startZ ||
            startX >= region.startX + region.sizeX ||
            startY >= region.startY + region.sizeY ||
            startZ >= region.startZ + region.sizeZ
        );
        
        if (overlaps) {
            std::cerr << "Error: Requested grid region overlaps with an existing region" << std::endl;
            return GridRegion(); // Return empty region on error
        }
    }
    
    // Create the new region
    GridRegion newRegion;
    newRegion.startX = startX;
    newRegion.startY = startY;
    newRegion.startZ = startZ;
    newRegion.sizeX = sizeX;
    newRegion.sizeY = sizeY;
    newRegion.sizeZ = sizeZ;
    newRegion.owner = component;
    
    // Add to allocated regions
    allocatedRegions.push_back(newRegion);
    
    std::cout << "Allocated grid region " 
              << "(" << startX << "," << startY << "," << startZ << ") "
              << "size " << sizeX << "x" << sizeY << "x" << sizeZ
              << " for component " << component->getName() << std::endl;
    
    return newRegion;
}

GridRegion SimulationManager::findAndAllocateGridRegion(
    std::shared_ptr<ComponentInterface> component,
    int sizeX, int sizeY, int sizeZ) {
    
    // Simple first-fit allocation strategy
    // In a real implementation, you might want a more sophisticated algorithm
    
    // Start from the origin and try to find a free region
    for (int z = 0; z <= params.gridSizeZ - sizeZ; z++) {
        for (int y = 0; y <= params.gridSizeY - sizeY; y++) {
            for (int x = 0; x <= params.gridSizeX - sizeX; x++) {
                // Check if this region would overlap with any existing region
                bool overlap = false;
                
                for (const auto& region : allocatedRegions) {
                    bool regionOverlap = !(
                        x + sizeX <= region.startX ||
                        y + sizeY <= region.startY ||
                        z + sizeZ <= region.startZ ||
                        x >= region.startX + region.sizeX ||
                        y >= region.startY + region.sizeY ||
                        z >= region.startZ + region.sizeZ
                    );
                    
                    if (regionOverlap) {
                        overlap = true;
                        break;
                    }
                }
                
                if (!overlap) {
                    // We found a free region, allocate it
                    return allocateGridRegion(component, x, y, z, sizeX, sizeY, sizeZ);
                }
            }
        }
    }
    
    // If we get here, no free region was found
    std::cerr << "Error: Could not find a free grid region of size "
              << sizeX << "x" << sizeY << "x" << sizeZ << std::endl;
    return GridRegion(); // Return empty region
}

void SimulationManager::releaseGridRegion(const GridRegion& region) {
    auto it = std::find_if(allocatedRegions.begin(), allocatedRegions.end(),
        [&region](const GridRegion& r) {
            return r.startX == region.startX &&
                   r.startY == region.startY &&
                   r.startZ == region.startZ &&
                   r.sizeX == region.sizeX &&
                   r.sizeY == region.sizeY &&
                   r.sizeZ == region.sizeZ;
        });
    
    if (it != allocatedRegions.end()) {
        std::cout << "Released grid region "
                  << "(" << region.startX << "," << region.startY << "," << region.startZ << ") "
                  << "size " << region.sizeX << "x" << region.sizeY << "x" << region.sizeZ
                  << std::endl;
        
        allocatedRegions.erase(it);
    } else {
        std::cerr << "Warning: Attempted to release non-existent grid region" << std::endl;
    }
}

glm::vec3 SimulationManager::gridToWorld(int gridX, int gridY, int gridZ) const {
    return glm::vec3(
        gridX * params.cellSize,
        gridY * params.cellSize,
        gridZ * params.cellSize
    );
}

void SimulationManager::worldToGrid(float worldX, float worldY, float worldZ,
                                   int& gridX, int& gridY, int& gridZ) const {
    gridX = static_cast<int>(worldX / params.cellSize);
    gridY = static_cast<int>(worldY / params.cellSize);
    gridZ = static_cast<int>(worldZ / params.cellSize);
}

bool SimulationManager::isPointAllocated(int x, int y, int z) const {
    for (const auto& region : allocatedRegions) {
        if (region.contains(x, y, z)) {
            return true;
        }
    }
    return false;
}

std::shared_ptr<ComponentInterface> SimulationManager::findComponentAtPoint(int x, int y, int z) const {
    for (const auto& region : allocatedRegions) {
        if (region.contains(x, y, z)) {
            return region.owner.lock(); // Convert weak_ptr to shared_ptr
        }
    }
    return nullptr;
}

const GridRegion* SimulationManager::findRegionAtPoint(int x, int y, int z) const {
    for (const auto& region : allocatedRegions) {
        if (region.contains(x, y, z)) {
            return &region;
        }
    }
    return nullptr;
}

void SimulationManager::localToGlobal(const GridRegion& region,
                                     int localX, int localY, int localZ,
                                     int& globalX, int& globalY, int& globalZ) const {
    globalX = region.startX + localX;
    globalY = region.startY + localY;
    globalZ = region.startZ + localZ;
}

bool SimulationManager::globalToLocal(const GridRegion& region,
                                     int globalX, int globalY, int globalZ,
                                     int& localX, int& localY, int& localZ) const {
    if (region.contains(globalX, globalY, globalZ)) {
        localX = globalX - region.startX;
        localY = globalY - region.startY;
        localZ = globalZ - region.startZ;
        return true;
    }
    return false;
}

} // namespace drumforge