#include "membrane.h"
#include <cmath>
#include <algorithm>

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
    , damping(damping) {
    
    // Initialize data arrays
    heights.resize(gridSize * gridSize, 0.0f);
    velocities.resize(gridSize * gridSize, 0.0f);
    
    // Initial state is a flat membrane (all heights = 0)
    reset();
}

void DrumMembrane::reset() {
    // Reset all heights and velocities to zero
    std::fill(heights.begin(), heights.end(), 0.0f);
    std::fill(velocities.begin(), velocities.end(), 0.0f);
}

void DrumMembrane::applyImpulse(float x, float y, float strength) {
    // To be implemented later
}

float DrumMembrane::getHeight(int x, int y) const {
    // Bounds checking
    if (x < 0 || x >= gridSize || y < 0 || y >= gridSize) {
        return 0.0f;
    }
    
    return heights[getIndex(x, y)];
}