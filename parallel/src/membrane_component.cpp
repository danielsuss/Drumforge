#include "membrane_component.h"
#include "membrane_kernels.cuh"
#include <stdexcept>
#include <iostream>
#include <cmath>
#include <vector>

// Forward declaration for VisualizationManager
#include "visualization_manager.h"

// Reference to global flag controlling CUDA-OpenGL interop
extern bool g_enableCudaGLInterop;

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
    , audioSamplePoint(0.5f, 0.5f)  // Default to center
    , audioGain(1.0f)
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
    // Get parameters from simulation manager
    const auto& params = simulationManager.getParameters();
    cellSize = params.cellSize;
    
    // Determine membrane dimensions based on radius
    // Add padding for boundary conditions
    const int padding = 4;  
    int membraneSize = static_cast<int>(2.0f * radius / cellSize) + 2 * padding;
    
    // Ensure membrane size is even for symmetry
    if (membraneSize % 2 != 0) {
        membraneSize++;
    }
    
    // Check if membrane would be too large for the grid
    const int maxGridDimension = std::min(params.gridSizeX, params.gridSizeY);
    if (membraneSize > maxGridDimension) {
        // Calculate the maximum radius that would fit
        float maxRadius = (maxGridDimension - 2 * padding) * cellSize / 2.0f;
        
        std::cout << "WARNING: Specified radius (" << radius << ") is too large for the grid." << std::endl;
        std::cout << "         Reducing radius to " << maxRadius << " to fit within grid." << std::endl;
        
        // Adjust radius to fit
        radius = maxRadius;
        
        // Recalculate membrane size with the new radius
        membraneSize = static_cast<int>(2.0f * radius / cellSize) + 2 * padding;
        if (membraneSize % 2 != 0) {
            membraneSize++;
        }
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
    
    // Set up OpenGL interop for visualization only if enabled
    if (g_enableCudaGLInterop) {
        try {
            // Check if CUDA-OpenGL interop is supported
            if (memoryManager.isGLInteropSupported()) {
                std::cout << "CUDA-OpenGL interoperability supported, setting up interop visualization" << std::endl;
                
                // We need to have the VAO/VBO created first before setting up interop
                // This might be done in the initializeVisualization method if not already created
                
                // For now, we'll defer the interop setup to the initializeVisualization method
            } else {
                std::cout << "CUDA-OpenGL interoperability not supported, visualization will use CPU-to-GPU transfers" << std::endl;
            }
        } catch (const CudaException& e) {
            std::cerr << "Warning: Failed to set up visualization interop: " << e.what() << std::endl;
            std::cerr << "Visualization will continue without GPU acceleration" << std::endl;
        }
    } else {
        std::cout << "CUDA-OpenGL interoperability disabled by configuration" << std::endl;
        std::cout << "Visualization will use CPU-to-GPU transfers" << std::endl;
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

    updateAudio(timestep);
}

void MembraneComponent::prepareForVisualization() {
    // Check if we can use CUDA-OpenGL interop (glInteropBuffer will be non-null if interop is set up)
    if (g_enableCudaGLInterop && glInteropBuffer && glInteropBuffer->isRegistered()) {
        try {
            // Using CUDA-OpenGL interop for direct GPU-to-GPU update

            // Map the buffer for CUDA access
            void* devicePtr = glInteropBuffer->map();
            
            // Convert to float3 pointer for vertex data
            float3* vertices = static_cast<float3*>(devicePtr);
            
            // CUDA kernel scale factor for visualization
            const float visualizationScale = 1.0f;
            
            // Call the CUDA kernel to update vertices directly on the GPU
            // This avoids the CPU-to-GPU transfer bottleneck
            updateVisualizationVertices(
                vertices,                  // OpenGL VBO mapped for CUDA access
                d_heights->get(),          // Current height field
                d_circleMask->get(),       // Circular membrane mask
                *kernelParams,             // Kernel parameters 
                visualizationScale         // Scale factor for visualization
            );
            
            // Unmap the buffer when done (makes it available for OpenGL again)
            glInteropBuffer->unmap();
            
            // No need to update the VAO or VBO - the data is already there
            std::cout << "Membrane visualization prepared using CUDA-OpenGL interop" << std::endl;
        }
        catch (const CudaException& e) {
            // Log error but continue - fallback to CPU-based update below
            std::cerr << "Error in CUDA-OpenGL interop: " << e.what() << std::endl;
            std::cerr << "Falling back to CPU-based visualization update" << std::endl;
            
            // Make sure the buffer is unmapped in case of partial failure
            if (glInteropBuffer->isMappedForCUDA()) {
                try {
                    glInteropBuffer->unmap();
                } 
                catch (...) {
                    // Ignore errors during cleanup
                }
            }
            
            // Continue to CPU fallback path
        }
    }
    
    // CPU-based fallback path or if interop is not available
    // Only use this if CUDA-OpenGL interop failed or is not available
    if (!g_enableCudaGLInterop || !glInteropBuffer || !glInteropBuffer->isRegistered()) {
        // Get the current heights from CUDA to CPU
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
            
            // std::cout << "Membrane visualization prepared using CPU-based update" << std::endl;
        }
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
        
        // CUDA-OpenGL Interop setup - Register the OpenGL vertex buffer with CUDA
        // This is the key part for interop functionality
        if (g_enableCudaGLInterop && memoryManager.isGLInteropSupported()) {
            try {
                // Register the VBO for CUDA access
                // First, we need the OpenGL buffer ID that was created by the VisualizationManager
                GLuint glBufferId;
                glBindVertexArray(vaoId);
                glGetVertexAttribIuiv(0, GL_VERTEX_ATTRIB_ARRAY_BUFFER_BINDING, &glBufferId);
                glBindVertexArray(0);
                
                // Now register this buffer with CUDA
                glInteropBuffer = memoryManager.registerGLBuffer(
                    glBufferId,
                    membraneWidth * membraneHeight,    // Total number of vertices
                    sizeof(float3),                    // Size of each vertex (x,y,z as float3)
                    cudaGraphicsRegisterFlagsWriteDiscard  // We only write to the buffer from CUDA
                );
                
                std::cout << "CUDA-OpenGL interop buffer successfully registered" << std::endl;
            }
            catch (const CudaException& e) {
                std::cerr << "Failed to register OpenGL buffer with CUDA: " << e.what() << std::endl;
                std::cerr << "Falling back to CPU-based visualization" << std::endl;
                glInteropBuffer = nullptr;  // Ensure we'll use the CPU path
            }
        } else {
            std::cout << "CUDA-OpenGL interop disabled, using CPU-based visualization" << std::endl;
            glInteropBuffer = nullptr;
        }
        
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

void MembraneComponent::setRadius(float newRadius) {
    // Don't allow negative or zero radius
    if (newRadius <= 0.0f) {
        std::cerr << "Warning: Cannot set radius to " << newRadius << ". Using minimum value of 0.1f instead." << std::endl;
        newRadius = 0.1f;
    }
    
    // Update the component's radius value
    radius = newRadius;
    
    // Update the kernel parameters
    kernelParams->radius = radius / cellSize;  // Convert physical radius to grid units
    kernelParams->radiusSquared = kernelParams->radius * kernelParams->radius;
    
    std::cout << "Membrane radius updated to " << radius << " (grid units: " << kernelParams->radius << ")" << std::endl;
    
    // Reinitialize the circular mask based on the new radius
    initializeCircleMask(d_circleMask->get(), *kernelParams);
    
    // Reset the membrane with the new parameters
    reset();
}

void MembraneComponent::setTension(float newTension) {
    // Don't allow negative or zero tension
    if (newTension <= 0.0f) {
        std::cerr << "Warning: Cannot set tension to " << newTension << ". Using minimum value of 0.1f instead." << std::endl;
        newTension = 0.1f;
    }
    
    // Update the component's tension value
    tension = newTension;
    
    // Update the kernel parameters
    kernelParams->tension = tension;
    kernelParams->waveSpeed = sqrt(tension);  // Wave speed derived from tension
    
    std::cout << "Membrane tension updated to " << tension << " (wave speed: " << kernelParams->waveSpeed << ")" << std::endl;
}

void MembraneComponent::setDamping(float newDamping) {
    // Ensure damping is non-negative
    if (newDamping < 0.0f) {
        std::cerr << "Warning: Cannot set damping to " << newDamping << ". Using minimum value of 0.0f instead." << std::endl;
        newDamping = 0.0f;
    }
    
    // Update the component's damping value
    damping = newDamping;
    
    // Update the kernel parameters
    kernelParams->damping = damping;
    
    std::cout << "Membrane damping updated to " << damping << std::endl;
}

void MembraneComponent::setAudioSamplePoint(float x, float y) {
    // Clamp to [0,1] range
    audioSamplePoint.x = std::max(0.0f, std::min(x, 1.0f));
    audioSamplePoint.y = std::max(0.0f, std::min(y, 1.0f));
}


void MembraneComponent::setAudioGain(float gain) {
    audioGain = gain;
}

void MembraneComponent::updateAudio(float timestep) {
    // Get reference to AudioManager
    AudioManager& audioManager = AudioManager::getInstance();
    
    // Only sample if recording is active
    if (audioManager.getIsRecording()) {
        // Convert normalized coordinates to grid coordinates
        int gridX = static_cast<int>(audioSamplePoint.x * membraneWidth);
        int gridY = static_cast<int>(audioSamplePoint.y * membraneHeight);
        
        // Get displacement at sample point
        float displacement = 0.0f;
        if (isInsideCircle(gridX, gridY)) {
            displacement = getHeight(gridX, gridY);
            
            // Apply gain
            displacement *= audioGain;
            
            // Process this time step with the current sample value
            audioManager.processAudioStep(timestep, displacement);
        }
    }
}

} // namespace drumforge