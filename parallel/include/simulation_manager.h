#ifndef DRUMFORGE_SIMULATION_MANAGER_H
#define DRUMFORGE_SIMULATION_MANAGER_H

#include "component_interface.h"
#include "cuda_memory_manager.h"
#include <vector>
#include <memory>
#include <string>
#include <unordered_map>
#include <glm/glm.hpp>

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
 * @brief Represents a region within the global grid
 * 
 * Used to track which parts of the global grid are allocated to specific components
 */
struct GridRegion {
    // Position in the global grid (bottom-left-front corner)
    int startX = 0;
    int startY = 0;
    int startZ = 0;
    
    // Size of the region
    int sizeX = 0;
    int sizeY = 0;
    int sizeZ = 0;
    
    // Owner component (optional)
    std::weak_ptr<ComponentInterface> owner;
    
    // Check if a global grid point is within this region
    bool contains(int x, int y, int z) const {
        return x >= startX && x < startX + sizeX &&
               y >= startY && y < startY + sizeY &&
               z >= startZ && z < startZ + sizeZ;
    }
};

/**
 * @brief Manager for coordinating all simulation components
 * 
 * This class orchestrates the simulation of all components,
 * handles their interactions, manages the simulation loop,
 * and maintains shared parameters like grid specifications.
 * It also coordinates the allocation of grid regions to components.
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
    
    // Grid region management
    std::vector<GridRegion> allocatedRegions;
    
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
    
    //--------------------------------------------------------------------------
    // Grid coordination functions
    //--------------------------------------------------------------------------
    
    // Allocate a region of the global grid for a component
    GridRegion allocateGridRegion(std::shared_ptr<ComponentInterface> component,
                                  int startX, int startY, int startZ,
                                  int sizeX, int sizeY, int sizeZ);
    
    // Find a free region in the global grid
    GridRegion findAndAllocateGridRegion(std::shared_ptr<ComponentInterface> component,
                                         int sizeX, int sizeY, int sizeZ);
    
    // Release a previously allocated grid region
    void releaseGridRegion(const GridRegion& region);
    
    // Get all allocated grid regions
    const std::vector<GridRegion>& getAllocatedRegions() const { return allocatedRegions; }
    
    // Convert global grid coordinates to world coordinates
    glm::vec3 gridToWorld(int gridX, int gridY, int gridZ) const;
    
    // Convert world coordinates to global grid coordinates
    void worldToGrid(float worldX, float worldY, float worldZ,
                    int& gridX, int& gridY, int& gridZ) const;
    
    // Check if a point in the global grid is within any allocated region
    bool isPointAllocated(int x, int y, int z) const;
    
    // Find which component owns a particular grid point
    std::shared_ptr<ComponentInterface> findComponentAtPoint(int x, int y, int z) const;
    
    // Find the grid region that contains a specific point
    const GridRegion* findRegionAtPoint(int x, int y, int z) const;
    
    // Convert from local component coordinates to global grid coordinates
    void localToGlobal(const GridRegion& region, 
                      int localX, int localY, int localZ,
                      int& globalX, int& globalY, int& globalZ) const;
    
    // Convert from global grid coordinates to local component coordinates
    bool globalToLocal(const GridRegion& region,
                      int globalX, int globalY, int globalZ,
                      int& localX, int& localY, int& localZ) const;
};

} // namespace drumforge

#endif // DRUMFORGE_SIMULATION_MANAGER_H