#ifndef DRUMFORGE_MEMBRANE_H
#define DRUMFORGE_MEMBRANE_H

#include <vector>

class DrumMembrane {
private:
    // Grid dimensions
    int gridSize;    // Number of points in each dimension (square grid)
    float radius;    // Radius in grid units from center to edge
    
    // Core data arrays
    std::vector<float> heights;     // Current height displacement at each point
    std::vector<float> velocities;  // Current vertical velocity at each point
    
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
};

#endif // DRUMFORGE_MEMBRANE_H