#ifndef DRUMFORGE_MEMBRANE_H
#define DRUMFORGE_MEMBRANE_H

#include <vector>
#include <glm/glm.hpp>

class DrumMembrane {
private:
    // Grid dimensions
    int gridSize;    // Number of points in each dimension (square grid)
    float radius;    // Radius in grid units from center to edge
    
    // Core data arrays
    std::vector<float> heights;     // Current height displacement at each point
    std::vector<float> velocities;  // Current vertical velocity at each point
    float prevTimestep;          // Previous time step value
    std::vector<float> prevHeights;  // Heights at previous time step for FDTD
    
    // Simulation parameters
    float tension;
    float damping;
    
    // Helper method to calculate 1D index from 2D coordinates
    int getIndex(int x, int y) const {
        return y * gridSize + x;
    }
    

public:
    // Constructor
    DrumMembrane(int size, float radius, float tension = 1.0f, float damping = 0.01f);
    
    // Basic manipulation methods
    void reset();  // Reset to flat state
    void applyImpulse(float x, float y, float strength);  // Apply impulse at position
    
    // Accessors
    float getHeight(int x, int y) const;
    int getGridSize() const { return gridSize; }
    float getRadius() const { return radius; }
    const std::vector<float>& getHeightData() const { return heights; }

    // Helper to determine if a point is inside the circular membrane boundary
    bool isInsideCircle(int x, int y) const;

    // Helper to generate vertices for rendering
    std::vector<glm::vec3> generateVertices() const;

    void updateSimulation(float timestep);  // Main simulation step function
    void setBoundaryConditions();          // Handle membrane boundary
    
    // Calculate a stable timestep for the membrane simulation
    float calculateStableTimestep() const {
        // Calculate wave propagation speed (c) for a membrane
        float c = sqrt(tension);  // Wave speed depends on tension
        
        // Grid spacing (normalized to 1.0 in our simulation)
        float dx = 1.0f;
        
        // CFL condition for 2D wave equation: dt <= dx / (c * sqrt(2))
        float cflTimestep = dx / (c * sqrt(2.0f));
        
        // Use 80% of CFL condition for safety
        return 0.8f * cflTimestep;
    }
};

#endif // DRUMFORGE_MEMBRANE_H