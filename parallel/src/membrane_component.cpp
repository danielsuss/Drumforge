#include "membrane_component.h"
#include "membrane_kernels.cuh"
#include <stdexcept>
#include <iostream>
#include <cmath>
#include <vector>

// Forward declaration for VisualizationManager
#include "visualization_manager.h"

namespace drumforge {

//-----------------------------------------------------------------------------
// Constructor & Destructor
//-----------------------------------------------------------------------------

MembraneComponent::MembraneComponent(const std::string& name, float radius, float tension, float damping)
    : memoryManager(CudaMemoryManager::getInstance())
    , simulationManager(SimulationManager::getInstance())
    , tension(tension)
    , damping(damping)
    , radius(radius)
    , membraneWidth(0)
    , membraneHeight(0)
    , cellSize(0.0f)
    , vaoId(0)
    , eboId(0)
    , name(name)
    , kernelParams(new MembraneKernelParams())
    , pendingImpulse{0.0f, 0.0f, 0.0f, false} {
    
    std::cout << "MembraneComponent '" << name << "' created" << std::endl;
}

MembraneComponent::~MembraneComponent() {
    std::cout << "MembraneComponent '" << name << "' destroyed" << std::endl;
    
    // Release grid region if allocated
    if (region.sizeX > 0 && region.sizeY > 0) {
        simulationManager.releaseGridRegion(region);
    }
    
    // OpenGL resources will be cleaned up by the VisualizationManager
    
    // CUDA buffers are automatically released by shared_ptr destruction
}

//-----------------------------------------------------------------------------
// ComponentInterface Implementation
//-----------------------------------------------------------------------------

void MembraneComponent::initialize() {
    std::cout << "Initializing MembraneComponent '" << name << "'..." << std::endl;
    
    // Get simulation parameters from SimulationManager
    const auto& params = simulationManager.getParameters();
    cellSize = params.cellSize;
    
    // Determine membrane dimensions based on radius
    // We need enough cells to cover the circular membrane
    // Add padding to ensure the membrane fits within the grid
    const int padding = 4;  // Extra cells around the membrane for boundary
    int membraneSize = static_cast<int>(2.0f * radius / cellSize) + 2 * padding;
    
    // Ensure membrane size is even for symmetry
    if (membraneSize % 2 != 0) {
        membraneSize++;
    }
    
    membraneWidth = membraneSize;
    membraneHeight = membraneSize;
    
    std::cout << "Membrane dimensions: " << membraneWidth << "x" << membraneHeight 
              << " (radius: " << radius << ", cell size: " << cellSize << ")" << std::endl;
    
    // Request grid region from SimulationManager
    region = simulationManager.findAndAllocateGridRegion(
        std::dynamic_pointer_cast<ComponentInterface>(shared_from_this()),
        membraneWidth, membraneHeight, 1  // Only need height=1 in Z dimension for membrane
    );
    
    if (region.sizeX == 0 || region.sizeY == 0) {
        throw std::runtime_error("Failed to allocate grid region for membrane");
    }
    
    std::cout << "Allocated grid region at (" << region.startX << "," << region.startY << "," 
              << region.startZ << ") of size " << region.sizeX << "x" << region.sizeY 
              << "x" << region.sizeZ << std::endl;
    
    // Allocate CUDA device buffers
    int totalElements = membraneWidth * membraneHeight;
    d_heights = memoryManager.allocateBuffer<float>(totalElements);
    d_prevHeights = memoryManager.allocateBuffer<float>(totalElements);
    d_velocities = memoryManager.allocateBuffer<float>(totalElements);
    d_circleMask = memoryManager.allocateBuffer<int>(totalElements);
    
    // Allocate host buffer for data retrieval when needed
    h_heights.resize(totalElements, 0.0f);
    
    // Set up kernel parameters
    kernelParams->membraneWidth = membraneWidth;
    kernelParams->membraneHeight = membraneHeight;
    kernelParams->radius = radius / cellSize;  // Convert physical radius to grid units
    kernelParams->radiusSquared = kernelParams->radius * kernelParams->radius;
    kernelParams->tension = tension;
    kernelParams->damping = damping;
    kernelParams->waveSpeed = sqrt(tension);  // Wave speed derived from tension
    kernelParams->centerX = membraneWidth / 2.0f;  // Center of the grid
    kernelParams->centerY = membraneHeight / 2.0f;
    
    // Initialize the circular mask
    initializeCircleMask(d_circleMask->get(), *kernelParams);
    
    // Reset the membrane to its initial flat state
    resetMembrane(d_heights->get(), d_prevHeights->get(), d_velocities->get(), 
                 d_circleMask->get(), *kernelParams);
    
    // Set up OpenGL interop for visualization
    try {
        // Check if CUDA-OpenGL interop is supported
        if (memoryManager.isGLInteropSupported()) {
            std::cout << "CUDA-OpenGL interoperability supported, visualization will be available" << std::endl;
        } else {
            std::cout << "CUDA-OpenGL interoperability not supported, visualization will be limited" << std::endl;
        }
    } catch (const CudaException& e) {
        std::cerr << "Warning: Failed to set up visualization: " << e.what() << std::endl;
        std::cerr << "Simulation will continue without visualization" << std::endl;
    }
    
    std::cout << "MembraneComponent '" << name << "' initialized successfully" << std::endl;
}

void MembraneComponent::update(float timestep) {
    // Calculate a stable timestep if the provided one is too large
    float stableTimestep = calculateStableTimestep();
    float safeTimestep = (timestep > stableTimestep) ? stableTimestep : timestep;
    
    // Check if there's a pending impulse to apply
    if (pendingImpulse.active) {
        applyImpulse(pendingImpulse.x, pendingImpulse.y, pendingImpulse.strength);
        pendingImpulse.active = false;  // Clear the pending flag
    }
    
    // Update the membrane simulation
    updateMembrane(d_heights->get(), d_prevHeights->get(), d_velocities->get(),
                  d_circleMask->get(), *kernelParams, safeTimestep);
}

void MembraneComponent::prepareForVisualization() {
    // For now, just create a simple static mesh for visualization
    // without using CUDA-OpenGL interop
    
    // Get the current heights - this will copy from CUDA to host
    const auto& heights = getHeights();
    
    // Create a CPU-side buffer for vertex data
    std::vector<float> vertices(membraneWidth * membraneHeight * 3);
    
    // Set up vertices for visualization
    const float visualizationScale = 1.0f; // Scale for better visibility
    for (int y = 0; y < membraneHeight; y++) {
        for (int x = 0; x < membraneWidth; x++) {
            int index = (y * membraneWidth + x) * 3;
            int dataIndex = y * membraneWidth + x;
            
            // Calculate position in world space
            float xPos = (x - membraneWidth / 2.0f) * cellSize;
            float yPos = (y - membraneHeight / 2.0f) * cellSize;
            float zPos = 0.0f; // Initially flat
            
            // Apply height if inside the circular membrane
            if (isInsideCircle(x, y)) {
                zPos = heights[dataIndex] * visualizationScale;
            }
            
            vertices[index] = xPos;
            vertices[index + 1] = yPos;
            vertices[index + 2] = zPos;
        }
    }
    
    // Update the OpenGL buffer with the new vertex data
    if (vaoId > 0) { // Only if VAO exists
        GLuint vbo;
        glBindVertexArray(vaoId);
        glGetVertexAttribIuiv(0, GL_VERTEX_ATTRIB_ARRAY_BUFFER_BINDING, &vbo);
        glBindBuffer(GL_ARRAY_BUFFER, vbo);
        glBufferSubData(GL_ARRAY_BUFFER, 0, vertices.size() * sizeof(float), vertices.data());
        glBindBuffer(GL_ARRAY_BUFFER, 0);
        glBindVertexArray(0);
    }
}

CouplingData MembraneComponent::getInterfaceData() {
    // For now, return an empty struct. This will be expanded when we implement coupling.
    return CouplingData();
}

void MembraneComponent::setCouplingData(const CouplingData& data) {
    // To be implemented when coupling with other components (e.g., air cavity)
}

std::string MembraneComponent::getName() const {
    return name;
}

float MembraneComponent::calculateStableTimestep() const {
    // Use the kernel helper to calculate a stable timestep
    return ::drumforge::calculateStableTimestep(*kernelParams);
}

//-----------------------------------------------------------------------------
// Visualization Methods
//-----------------------------------------------------------------------------

void MembraneComponent::initializeVisualization(VisualizationManager& visManager) {
    // Check if already initialized
    if (vaoId != 0) {
        return;
    }
    
    std::cout << "Initializing visualization for MembraneComponent '" << name << "'..." << std::endl;
    
    try {
        // Initialize vertices for a simple grid
        std::vector<float> vertices(membraneWidth * membraneHeight * 3);
        for (int y = 0; y < membraneHeight; y++) {
            for (int x = 0; x < membraneWidth; x++) {
                int index = (y * membraneWidth + x) * 3;
                
                // Calculate position in world space
                float xPos = (x - membraneWidth / 2.0f) * cellSize;
                float yPos = (y - membraneHeight / 2.0f) * cellSize;
                float zPos = 0.0f; // Initially flat
                
                vertices[index] = xPos;
                vertices[index + 1] = yPos;
                vertices[index + 2] = zPos;
            }
        }
        
        // Create a vertex buffer with initial data
        unsigned int vbo = visManager.createVertexBuffer(
            vertices.size() * sizeof(float), 
            vertices.data()
        );
        
        // Create Vertex Array Object
        vaoId = visManager.createVertexArray();
        
        // Configure vertex attributes
        visManager.configureVertexAttributes(vaoId, vbo, 0, 3, 3 * sizeof(float), 0);
        
        // Generate indices for wireframe rendering - ONLY within membrane radius
        std::vector<unsigned int> indices;
        
        // Generate row lines only for points within the membrane radius
        for (int y = 0; y < membraneHeight; y++) {
            for (int x = 0; x < membraneWidth - 1; x++) {
                // Only add lines if both points are inside the circle
                if (isInsideCircle(x, y) && isInsideCircle(x + 1, y)) {
                    indices.push_back(y * membraneWidth + x);
                    indices.push_back(y * membraneWidth + x + 1);
                }
            }
        }
        
        // Generate column lines only for points within the membrane radius
        for (int x = 0; x < membraneWidth; x++) {
            for (int y = 0; y < membraneHeight - 1; y++) {
                // Only add lines if both points are inside the circle
                if (isInsideCircle(x, y) && isInsideCircle(x, y + 1)) {
                    indices.push_back(y * membraneWidth + x);
                    indices.push_back((y + 1) * membraneWidth + x);
                }
            }
        }
        
        // Create Element Buffer Object
        eboId = visManager.createIndexBuffer(indices.data(), indices.size() * sizeof(unsigned int));
        
        std::cout << "Visualization for MembraneComponent '" << name << "' initialized successfully" << std::endl;
    }
    catch (const std::exception& e) {
        std::cerr << "Error initializing visualization: " << e.what() << std::endl;
        // Don't rethrow - we want to continue even if visualization fails
    }
}

void MembraneComponent::visualize(VisualizationManager& visManager) {
    // Prepare data for visualization first (update vertex buffer via CUDA)
    prepareForVisualization();
    
    // Tell the visualization manager to render this component
    // The membrane uses a simple wireframe with a blue color
    visManager.renderWireframe(
        vaoId,                  // VAO containing the membrane geometry
        eboId,                  // EBO containing the wireframe indices
        getIndexCount(),        // Number of indices to draw
        glm::vec3(0.2f, 0.3f, 0.8f)  // Blue color for the membrane
    );
}

int MembraneComponent::getIndexCount() const {
    // If we have no EBO yet, return 0
    if (eboId == 0) {
        return 0;
    }
    
    // Get the actual count from OpenGL
    GLint count;
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, eboId);
    glGetBufferParameteriv(GL_ELEMENT_ARRAY_BUFFER, GL_BUFFER_SIZE, &count);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
    
    // Convert byte count to index count (using unsigned int indices)
    return count / sizeof(unsigned int);
}

//-----------------------------------------------------------------------------
// Membrane-specific Methods
//-----------------------------------------------------------------------------

void MembraneComponent::applyImpulse(float x, float y, float strength) {
    // Apply an impulse at the normalized position (x, y) with the given strength
    ::drumforge::applyImpulse(d_heights->get(), d_circleMask->get(), *kernelParams, x, y, strength);
    
    std::cout << "Applied impulse at position (" << x << "," << y 
              << ") with strength " << strength << std::endl;
}

void MembraneComponent::reset() {
    // Reset the membrane to its initial flat state
    resetMembrane(d_heights->get(), d_prevHeights->get(), d_velocities->get(),
                 d_circleMask->get(), *kernelParams);
    
    std::cout << "Membrane reset to initial state" << std::endl;
}

const std::vector<float>& MembraneComponent::getHeights() const {
    // Copy the current heights from device to host for CPU access
    // This is only needed for CPU-side operations, not for simulation
    
    // Cast away constness for this operation - the function is logically const
    // even though it updates the cache of height values
    auto& nonConstThis = const_cast<MembraneComponent&>(*this);
    auto& nonConstHeights = const_cast<std::vector<float>&>(h_heights);
    
    d_heights->copyToHost(nonConstHeights.data());
    return h_heights;
}

float MembraneComponent::getHeight(int x, int y) const {
    // Check bounds
    if (x < 0 || x >= membraneWidth || y < 0 || y >= membraneHeight) {
        return 0.0f;
    }
    
    // Copy the current heights from device to host if necessary
    // This is inefficient for many individual queries; better to use getHeights() once
    const_cast<MembraneComponent*>(this)->getHeights();
    
    // Return the height at the specified position
    return h_heights[y * membraneWidth + x];
}

bool MembraneComponent::isInsideCircle(int x, int y) const {
    // Check if a point is inside the circular membrane boundary
    float dx = x - membraneWidth / 2.0f;
    float dy = y - membraneHeight / 2.0f;
    float distanceSquared = dx * dx + dy * dy;
    
    return distanceSquared <= (kernelParams->radius * kernelParams->radius);
}

} // namespace drumforge