#include "membrane.h"
#include <cmath>
#include <algorithm>
#include <glm/glm.hpp>

bool DrumMembrane::isInsideCircle(int x, int y) const {
    // Calculate grid center
    float centerX = (gridSize - 1) / 2.0f;
    float centerY = (gridSize - 1) / 2.0f;
    
    // Calculate squared distance from center
    float dx = x - centerX;
    float dy = y - centerY;
    float distanceSquared = dx * dx + dy * dy;
    
    // Check if point is inside circle
    return distanceSquared <= (radius * radius);
}

DrumMembrane::DrumMembrane(int size, float radius, float tension, float damping)
    : gridSize(size)
    , radius(radius)
    , tension(tension)
    , damping(damping) 
    , prevTimestep(0.0f) {
    
    // Initialize data arrays
    heights.resize(gridSize * gridSize, 0.0f);
    velocities.resize(gridSize * gridSize, 0.0f);
    prevHeights.resize(gridSize * gridSize, 0.0f);
    
    // Initial state is a flat membrane (all heights = 0)
    reset();
}

void DrumMembrane::reset() {
    // Reset all heights and velocities to zero
    std::fill(heights.begin(), heights.end(), 0.0f);
    std::fill(velocities.begin(), velocities.end(), 0.0f);
    std::fill(prevHeights.begin(), prevHeights.end(), 0.0f);
}

void DrumMembrane::setBoundaryConditions() {
    // For a circular membrane, we set the boundary to zero (fixed edge)
    for (int y = 0; y < gridSize; y++) {
        for (int x = 0; x < gridSize; x++) {
            // If point is outside circle or on the edge, set height to zero
            if (!isInsideCircle(x, y)) {
                heights[getIndex(x, y)] = 0.0f;
                prevHeights[getIndex(x, y)] = 0.0f;
                velocities[getIndex(x, y)] = 0.0f;
            }
        }
    }
}

// Implement the main simulation step
void DrumMembrane::updateSimulation(float timestep) {
    // Wave equation parameters
    float c = sqrt(tension);  // Wave propagation speed (simplified)
    
    // First time step handling
    if (prevTimestep <= 0.0f) {
        prevTimestep = timestep;
        // For first step, we copy current heights to prevHeights
        prevHeights = heights;
        return;
    }
    
    // Create a temporary array for the new heights
    std::vector<float> newHeights = heights;  // Start with current heights
    
    // CFL stability condition check
    float dx = 1.0f;  // Grid spacing (normalized)
    float maxTimestep = 0.5f * dx / c;  // Maximum stable time step
    
    if (timestep > maxTimestep) {
        timestep = maxTimestep;  // Clamp to stable value
    }
    
    // FDTD coefficient
    float coef = (c * timestep / dx) * (c * timestep / dx);
    
    // Update interior points using FDTD method
    for (int y = 1; y < gridSize - 1; y++) {
        for (int x = 1; x < gridSize - 1; x++) {
            if (isInsideCircle(x, y)) {
                int idx = getIndex(x, y);
                
                // Gather neighboring points
                float uCenter = heights[idx];
                float uLeft = heights[getIndex(x-1, y)];
                float uRight = heights[getIndex(x+1, y)];
                float uTop = heights[getIndex(x, y-1)];
                float uBottom = heights[getIndex(x, y+1)];
                
                // Discrete Laplacian
                float laplacian = uLeft + uRight + uTop + uBottom - 4.0f * uCenter;
                
                // FDTD update equation: u(t+dt) = 2*u(t) - u(t-dt) + coef * laplacian
                newHeights[idx] = 2.0f * uCenter - prevHeights[idx] + coef * laplacian;
                
                // Apply damping
                newHeights[idx] = newHeights[idx] * (1.0f - damping);
            }
        }
    }
    
    // Update state for next iteration
    prevHeights = heights;
    heights = newHeights;
    
    // Apply boundary conditions
    setBoundaryConditions();
    
    // Store current timestep
    prevTimestep = timestep;
}

void DrumMembrane::applyImpulse(float x, float y, float strength) {
    // Convert normalized coordinates [0,1] to grid coordinates
    int gridX = static_cast<int>(x * gridSize);
    int gridY = static_cast<int>(y * gridSize);
    
    // Check bounds
    if (gridX < 0 || gridX >= gridSize || gridY < 0 || gridY >= gridSize) {
        return;
    }
    
    // Apply impulse in a small region around the hit point
    const int radius = 5;  // Size of impulse region
    
    for (int dy = -radius; dy <= radius; dy++) {
        for (int dx = -radius; dx <= radius; dx++) {
            int ix = gridX + dx;
            int iy = gridY + dy;
            
            // Check bounds and if inside circle
            if (ix >= 0 && ix < gridSize && iy >= 0 && iy < gridSize && isInsideCircle(ix, iy)) {
                // Calculate distance from impulse center
                float dist = sqrt(dx*dx + dy*dy);
                
                // Apply Gaussian-shaped impulse
                if (dist <= radius) {
                    float factor = strength * exp(-dist*dist / (radius/2.0f));
                    int idx = getIndex(ix, iy);
                    heights[idx] += factor;
                }
            }
        }
    }
}

float DrumMembrane::getHeight(int x, int y) const {
    // Bounds checking
    if (x < 0 || x >= gridSize || y < 0 || y >= gridSize) {
        return 0.0f;
    }
    
    return heights[getIndex(x, y)];
}

std::vector<glm::vec3> DrumMembrane::generateVertices() const {
    std::vector<glm::vec3> vertices;
    
    // Reserve space for efficiency
    vertices.reserve(gridSize * gridSize);
    
    // Loop through the grid
    for (int y = 0; y < gridSize; y++) {
        for (int x = 0; x < gridSize; x++) {
            // Use raw grid coordinates without normalization
            float nx = static_cast<float>(x);
            float ny = static_cast<float>(y);
            
            // Get height at this point (will be 0 initially)
            float nz = getHeight(x, y);
            
            // Add vertex to the list
            vertices.push_back(glm::vec3(nx, ny, nz));
        }
    }
    
    return vertices;
}