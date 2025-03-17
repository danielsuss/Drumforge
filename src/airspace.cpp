#include "airspace.h"
#include <iostream>

AirSpace::AirSpace(int sizeX, int sizeY, int sizeZ, float cellSize)
    : sizeX(sizeX)
    , sizeY(sizeY)
    , sizeZ(sizeZ)
    , cellSize(cellSize)
    , posX(0.0f)
    , posY(0.0f)
    , posZ(0.0f) {
    
    std::cout << "AirSpace initialized with dimensions: " 
              << sizeX << "x" << sizeY << "x" << sizeZ << std::endl;
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