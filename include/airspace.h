#ifndef DRUMFORGE_AIRSPACE_H
#define DRUMFORGE_AIRSPACE_H

#include <vector>
#include <glm/glm.hpp>

class AirSpace {
private:
    // Grid dimensions
    int sizeX, sizeY, sizeZ;  // Number of cells in each dimension
    float cellSize;           // Size of each cell in world units
    
    // Position in world coordinates
    float posX, posY, posZ;
    
    // Helper method to calculate 1D index from 3D coordinates
    int getIndex(int x, int y, int z) const {
        return z * sizeX * sizeY + y * sizeX + x;
    }
    
public:
    // Constructor
    AirSpace(int sizeX, int sizeY, int sizeZ, float cellSize = 1.0f);
    
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
    
    // Calculate world coordinates from grid coordinates
    glm::vec3 gridToWorld(int x, int y, int z) const;
    
    // Calculate grid coordinates from world coordinates
    void worldToGrid(float worldX, float worldY, float worldZ, 
                    int& gridX, int& gridY, int& gridZ) const;
};

#endif // DRUMFORGE_AIRSPACE_H