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
    
    // Default damping value is likely too high - start with a much smaller value
    // Typical values for musical membranes are around 0.01 to 0.001
    if (damping > 0.1f) {
        this->damping = 0.01f;
    }
    
    // Initial state is a flat membrane (all heights = 0)
    reset();
}

void DrumMembrane::reset() {
    // Reset all heights and velocities to zero
    std::fill(heights.begin(), heights.end(), 0.0f);
    std::fill(velocities.begin(), velocities.end(), 0.0f);
    std::fill(prevHeights.begin(), prevHeights.end(), 0.0f);
    prevTimestep = 0.0f;
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

void DrumMembrane::updateSimulation(float timestep) {
    // Wave equation parameters
    float c = sqrt(tension);  // Wave propagation speed
    float dx = 1.0f;  // Grid spacing (normalized)
    
    // CFL stability condition check for 2D wave equation
    float maxTimestep = dx / (c * sqrt(2.0f));
    
    // Clamp timestep to maintain stability
    float safeTimestep = (timestep > maxTimestep) ? maxTimestep : timestep;
    
    // FDTD coefficient using the safe timestep
    float coef = c * c * safeTimestep * safeTimestep / (dx * dx);
    
    // Create a temporary array for the new heights
    std::vector<float> newHeights = heights;  // Start with current heights
    
    // First time step special handling
    if (prevTimestep <= 0.0f) {
        // For the first time step, we need special handling
        // We use the wave equation and current velocities
        
        for (int y = 1; y < gridSize - 1; y++) {
            for (int x = 1; x < gridSize - 1; x++) {
                if (isInsideCircle(x, y)) {
                    int idx = getIndex(x, y);
                    
                    // Gather neighboring points for Laplacian
                    float uCenter = heights[idx];
                    float uLeft = heights[getIndex(x-1, y)];
                    float uRight = heights[getIndex(x+1, y)];
                    float uTop = heights[getIndex(x, y-1)];
                    float uBottom = heights[getIndex(x, y+1)];
                    
                    // Discrete Laplacian
                    float laplacian = uLeft + uRight + uTop + uBottom - 4.0f * uCenter;
                    
                    // Calculate acceleration from Laplacian and damping
                    float acceleration = c * c * laplacian / (dx * dx) - damping * velocities[idx];
                    
                    // Initialize previous heights based on current heights and velocities
                    // Using backward Euler approximation: u(t-dt) = u(t) - dt*v(t)
                    prevHeights[idx] = heights[idx] - safeTimestep * velocities[idx];
                    
                    // For first step: u(t+dt) = u(t) + dt*v(t) + 0.5*dt²*a(t)
                    // Using Taylor expansion
                    newHeights[idx] = uCenter + safeTimestep * velocities[idx] + 
                                     0.5f * safeTimestep * safeTimestep * acceleration;
                }
            }
        }
    }
    else {
        // Normal FDTD update for subsequent steps
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
                    
                    // FDTD update equation: u(t+dt) = 2*u(t) - u(t-dt) + c²*(dt/dx)²*∇²u(t)
                    // This is the standard leapfrog update for the wave equation
                    newHeights[idx] = 2.0f * uCenter - prevHeights[idx] + coef * laplacian;
                    
                    // Apply damping term - the standard formulation for damped wave equations
                    // uses a damping term proportional to the time derivative of u
                    // This can be approximated with central differences
                    float damping_term = damping * (heights[idx] - prevHeights[idx]) / safeTimestep;
                    
                    // Apply damping term to the update equation
                    // Note that with the standard wave equation damping, we subtract a term
                    // proportional to the first time derivative
                    newHeights[idx] -= damping_term * safeTimestep;
                }
            }
        }
    }
    
    // Update state for next iteration
    prevHeights = heights;
    heights = newHeights;
    
    // Apply boundary conditions
    setBoundaryConditions();
    
    // Store current timestep
    prevTimestep = safeTimestep;
    
    // Update velocities based on central difference
    for (int y = 1; y < gridSize - 1; y++) {
        for (int x = 1; x < gridSize - 1; x++) {
            if (isInsideCircle(x, y)) {
                int idx = getIndex(x, y);
                // v(t) = (u(t+dt) - u(t-dt)) / (2*dt)
                velocities[idx] = (heights[idx] - prevHeights[idx]) / (2.0f * safeTimestep);
            }
        }
    }
}

void DrumMembrane::applyImpulse(float x, float y, float strength) {
    // Convert normalized coordinates [0,1] to grid coordinates
    int gridX = static_cast<int>(x * gridSize);
    int gridY = static_cast<int>(y * gridSize);
    
    // Check bounds
    if (gridX < 0) gridX = 0;
    if (gridX >= gridSize) gridX = gridSize - 1;
    if (gridY < 0) gridY = 0;
    if (gridY >= gridSize) gridY = gridSize - 1;
    
    // Apply impulse in a small region around the hit point
    const int impulseRadius = 5;  // Size of impulse region
    
    for (int dy = -impulseRadius; dy <= impulseRadius; dy++) {
        for (int dx = -impulseRadius; dx <= impulseRadius; dx++) {
            int ix = gridX + dx;
            int iy = gridY + dy;
            
            // Check bounds and if inside circle
            if (ix >= 0 && ix < gridSize && iy >= 0 && iy < gridSize && isInsideCircle(ix, iy)) {
                // Calculate distance from impulse center
                float dist = sqrt(dx*dx + dy*dy);
                
                // Apply Gaussian-shaped impulse
                if (dist <= impulseRadius) {
                    float factor = strength * exp(-dist*dist / (impulseRadius/2.0f));
                    int idx = getIndex(ix, iy);
                    // Use negative factor to make the membrane go down instead of up
                    heights[idx] -= factor;  // Changed from += to -=
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
            
            // Get height at this point
            float nz = getHeight(x, y);
            
            // Add vertex to the list
            vertices.push_back(glm::vec3(nx, ny, nz));
        }
    }
    
    return vertices;
}