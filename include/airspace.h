#ifndef DRUMFORGE_AIRSPACE_H
#define DRUMFORGE_AIRSPACE_H

#include <vector>
#include <glm/glm.hpp>
#include <memory>

class AirSpace {
private:
    // Grid dimensions
    int sizeX, sizeY, sizeZ;  // Number of cells in each dimension
    float cellSize;           // Size of each cell in world units
    
    // Position in world coordinates
    float posX, posY, posZ;
    
    // Acoustic simulation parameters
    float speedOfSound;       // Speed of sound in air (m/s)
    float density;            // Density of air (kg/m^3)
    float timestep;           // Simulation time step (s)
    float dampingCoefficient; // Damping coefficient for stability and absorption
    
    // Acoustic field data
    std::vector<float> pressure;       // Current pressure field
    std::vector<float> prevPressure;   // Previous pressure field
    std::vector<float> velocityX;      // X-component of velocity (staggered grid)
    std::vector<float> velocityY;      // Y-component of velocity (staggered grid)
    std::vector<float> velocityZ;      // Z-component of velocity (staggered grid)
    
    // Helper methods for indexing
    int getIndex(int x, int y, int z) const {
        return z * sizeX * sizeY + y * sizeX + x;
    }
    
    int getVelocityXIndex(int x, int y, int z) const {
        return z * (sizeX + 1) * sizeY + y * (sizeX + 1) + x;
    }
    
    int getVelocityYIndex(int x, int y, int z) const {
        return z * sizeX * (sizeY + 1) + y * sizeX + x;
    }
    
    int getVelocityZIndex(int x, int y, int z) const {
        return z * sizeX * sizeY + y * sizeX + x;
    }
    
public:
    // Constructor
    AirSpace(int sizeX, int sizeY, int sizeZ, float cellSize = 1.0f);
    
    // Destructor
    ~AirSpace() = default;
    
    // Initialize acoustic simulation
    void initializeSimulation();
    
    // Calculate stable timestep based on CFL condition
    float calculateStableTimestep() const;
    
    // Set the actual timestep to use (might be determined by membrane simulation)
    void setTimestep(float dt);
    
    // Main simulation step
    void updateSimulation();
    
    // Apply boundary conditions
    void applyBoundaryConditions();
    
    // Utility to add an impulse (pressure disturbance) at a point
    void addPressureImpulse(float x, float y, float z, float strength, float radius);
    
    // Virtual microphone: get pressure at a specific position
    float getPressureAt(float x, float y, float z) const;
    
    // Set position in world coordinates
    void setPosition(float x, float y, float z);
    
    // Getters
    int getSizeX() const { return sizeX; }
    int getSizeY() const { return sizeY; }
    int getSizeZ() const { return sizeZ; }
    float getCellSize() const { return cellSize; }
    float getPosX() const { return posX; }
    float getPosY() const { return posY; }
    float getPosZ() const { return posZ; }
    float getSpeedOfSound() const { return speedOfSound; }
    float getDensity() const { return density; }
    float getTimestep() const { return timestep; }
    
    // Calculate world coordinates from grid coordinates
    glm::vec3 gridToWorld(int x, int y, int z) const;
    
    // Calculate grid coordinates from world coordinates
    void worldToGrid(float worldX, float worldY, float worldZ, 
                    int& gridX, int& gridY, int& gridZ) const;
                    
    // Get pressure field for visualization
    const std::vector<float>& getPressureField() const { return pressure; }
};

#endif // DRUMFORGE_AIRSPACE_H