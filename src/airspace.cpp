#include "airspace.h"
#include <iostream>
#include <cmath>
#include <algorithm>

AirSpace::AirSpace(int sizeX, int sizeY, int sizeZ, float cellSize)
    : sizeX(sizeX)
    , sizeY(sizeY)
    , sizeZ(sizeZ)
    , cellSize(cellSize)
    , posX(0.0f)
    , posY(0.0f)
    , posZ(0.0f)
    , speedOfSound(343.0f)
    , density(1.2f)
    , timestep(0.0f)
    , dampingCoefficient(0.001f) {
    
    std::cout << "AirSpace initialized with dimensions: " 
              << sizeX << "x" << sizeY << "x" << sizeZ << std::endl;
              
    // Initialize the acoustic simulation
    initializeSimulation();
}

void AirSpace::initializeSimulation() {
    // Calculate stable timestep based on CFL condition
    timestep = calculateStableTimestep();
    std::cout << "AirSpace stable timestep: " << timestep << " seconds" << std::endl;
    
    // Allocate memory for acoustic fields
    pressure.resize(sizeX * sizeY * sizeZ, 0.0f);
    prevPressure.resize(sizeX * sizeY * sizeZ, 0.0f);
    
    // Velocity fields on staggered grids
    // X-velocity: One more in X direction
    velocityX.resize((sizeX + 1) * sizeY * sizeZ, 0.0f);
    
    // Y-velocity: One more in Y direction
    velocityY.resize(sizeX * (sizeY + 1) * sizeZ, 0.0f);
    
    // Z-velocity: One more in Z direction
    velocityZ.resize(sizeX * sizeY * (sizeZ + 1), 0.0f);
    
    std::cout << "AirSpace acoustic simulation initialized" << std::endl;
}

float AirSpace::calculateStableTimestep() const {
    // Calculate Courant number for 3D FDTD (must be < 1/sqrt(3) for stability)
    const float courantNumber = 0.5f;
    
    // Calculate stable timestep
    return courantNumber * cellSize / speedOfSound;
}

void AirSpace::setTimestep(float dt) {
    float stableTimestep = calculateStableTimestep();
    
    if (dt > stableTimestep) {
        std::cout << "Warning: Requested timestep " << dt << " exceeds stable timestep " 
                  << stableTimestep << ". Using stable timestep instead." << std::endl;
        timestep = stableTimestep;
    } else {
        timestep = dt;
    }
}

void AirSpace::updateSimulation() {
    // 1. Update velocity components based on pressure gradients
    for (int z = 0; z < sizeZ; z++) {
        for (int y = 0; y < sizeY; y++) {
            for (int x = 0; x < sizeX + 1; x++) {
                // Skip boundary if at edge
                if (x == 0 || x == sizeX) continue;
                
                // X-velocity update (pressure gradient in x direction)
                float pressureGradientX = (pressure[getIndex(x, y, z)] - pressure[getIndex(x-1, y, z)]) / cellSize;
                velocityX[getVelocityXIndex(x, y, z)] -= (timestep / density) * pressureGradientX;
            }
        }
    }
    
    for (int z = 0; z < sizeZ; z++) {
        for (int y = 0; y < sizeY + 1; y++) {
            for (int x = 0; x < sizeX; x++) {
                // Skip boundary if at edge
                if (y == 0 || y == sizeY) continue;
                
                // Y-velocity update (pressure gradient in y direction)
                float pressureGradientY = (pressure[getIndex(x, y, z)] - pressure[getIndex(x, y-1, z)]) / cellSize;
                velocityY[getVelocityYIndex(x, y, z)] -= (timestep / density) * pressureGradientY;
            }
        }
    }
    
    for (int z = 0; z < sizeZ + 1; z++) {
        for (int y = 0; y < sizeY; y++) {
            for (int x = 0; x < sizeX; x++) {
                // Skip boundary if at edge
                if (z == 0 || z == sizeZ) continue;
                
                // Z-velocity update (pressure gradient in z direction)
                float pressureGradientZ = (pressure[getIndex(x, y, z)] - pressure[getIndex(x, y, z-1)]) / cellSize;
                velocityZ[getVelocityZIndex(x, y, z)] -= (timestep / density) * pressureGradientZ;
            }
        }
    }
    
    // 2. Store current pressure for next step
    prevPressure = pressure;
    
    // 3. Update pressure based on velocity divergence
    for (int z = 0; z < sizeZ; z++) {
        for (int y = 0; y < sizeY; y++) {
            for (int x = 0; x < sizeX; x++) {
                // Calculate velocity divergence
                float velocityDivergence = 
                    (velocityX[getVelocityXIndex(x+1, y, z)] - velocityX[getVelocityXIndex(x, y, z)]) / cellSize +
                    (velocityY[getVelocityYIndex(x, y+1, z)] - velocityY[getVelocityYIndex(x, y, z)]) / cellSize +
                    (velocityZ[getVelocityZIndex(x, y, z+1)] - velocityZ[getVelocityZIndex(x, y, z)]) / cellSize;
                
                // Update pressure using acoustic wave equation
                float pressureChange = -density * speedOfSound * speedOfSound * timestep * velocityDivergence;
                
                // Apply damping for stability
                pressure[getIndex(x, y, z)] = prevPressure[getIndex(x, y, z)] + pressureChange - 
                                             dampingCoefficient * prevPressure[getIndex(x, y, z)];
            }
        }
    }
    
    // 4. Apply boundary conditions
    applyBoundaryConditions();
}

void AirSpace::applyBoundaryConditions() {
    // Apply rigid boundary conditions (set normal velocities to zero at boundaries)
    
    // X-direction boundaries (x = 0 and x = sizeX)
    for (int z = 0; z < sizeZ; z++) {
        for (int y = 0; y < sizeY; y++) {
            velocityX[getVelocityXIndex(0, y, z)] = 0.0f;
            velocityX[getVelocityXIndex(sizeX, y, z)] = 0.0f;
        }
    }
    
    // Y-direction boundaries (y = 0 and y = sizeY)
    for (int z = 0; z < sizeZ; z++) {
        for (int x = 0; x < sizeX; x++) {
            velocityY[getVelocityYIndex(x, 0, z)] = 0.0f;
            velocityY[getVelocityYIndex(x, sizeY, z)] = 0.0f;
        }
    }
    
    // Z-direction boundaries (z = 0 and z = sizeZ)
    for (int y = 0; y < sizeY; y++) {
        for (int x = 0; x < sizeX; x++) {
            velocityZ[getVelocityZIndex(x, y, 0)] = 0.0f;
            velocityZ[getVelocityZIndex(x, y, sizeZ)] = 0.0f;
        }
    }
}

void AirSpace::addPressureImpulse(float x, float y, float z, float strength, float radius) {
    // Convert world coordinates to grid coordinates
    int gridX, gridY, gridZ;
    worldToGrid(x, y, z, gridX, gridY, gridZ);
    
    // Calculate radius in grid units
    int gridRadius = static_cast<int>(radius / cellSize);
    if (gridRadius < 1) gridRadius = 1;
    
    // Add Gaussian pressure impulse centered at the specified point
    for (int dz = -gridRadius; dz <= gridRadius; dz++) {
        for (int dy = -gridRadius; dy <= gridRadius; dy++) {
            for (int dx = -gridRadius; dx <= gridRadius; dx++) {
                int ix = gridX + dx;
                int iy = gridY + dy;
                int iz = gridZ + dz;
                
                // Check if the point is within grid bounds
                if (ix >= 0 && ix < sizeX && iy >= 0 && iy < sizeY && iz >= 0 && iz < sizeZ) {
                    // Calculate distance from impulse center
                    float dist = std::sqrt(dx*dx + dy*dy + dz*dz);
                    
                    // Apply Gaussian-shaped impulse
                    if (dist <= gridRadius) {
                        float factor = strength * std::exp(-dist*dist / (gridRadius/2.0f));
                        pressure[getIndex(ix, iy, iz)] += factor;
                    }
                }
            }
        }
    }
}

float AirSpace::getPressureAt(float x, float y, float z) const {
    // Convert world coordinates to grid coordinates
    int gridX, gridY, gridZ;
    worldToGrid(x, y, z, gridX, gridY, gridZ);
    
    // Check if the point is within grid bounds
    if (gridX >= 0 && gridX < sizeX && gridY >= 0 && gridY < sizeY && gridZ >= 0 && gridZ < sizeZ) {
        return pressure[getIndex(gridX, gridY, gridZ)];
    }
    
    // Return zero pressure if out of bounds
    return 0.0f;
}

void AirSpace::setPosition(float x, float y, float z) {
    posX = x;
    posY = y;
    posZ = z;
}

glm::vec3 AirSpace::gridToWorld(int x, int y, int z) const {
    return glm::vec3(
        posX + x * cellSize,
        posY + y * cellSize,
        posZ + z * cellSize
    );
}

void AirSpace::worldToGrid(float worldX, float worldY, float worldZ, 
                          int& gridX, int& gridY, int& gridZ) const {
    gridX = static_cast<int>((worldX - posX) / cellSize);
    gridY = static_cast<int>((worldY - posY) / cellSize);
    gridZ = static_cast<int>((worldZ - posZ) / cellSize);
}