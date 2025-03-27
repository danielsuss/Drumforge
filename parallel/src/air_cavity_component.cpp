#include "air_cavity_component.h"
#include "air_cavity_kernels.cuh"
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

AirCavityComponent::AirCavityComponent(const std::string& name, float speedOfSound, float density, float dampingCoefficient)
    : memoryManager(CudaMemoryManager::getInstance())
    , simulationManager(SimulationManager::getInstance())
    , speedOfSound(speedOfSound)
    , density(density)
    , dampingCoefficient(dampingCoefficient)
    , cavityWidth(0)
    , cavityHeight(0)
    , cavityDepth(0)
    , cellSize(0.0f)
    , vaoId(0)
    , eboId(0)
    , currentSliceZ(0)
    , name(name)
    , kernelParams(new AirCavityKernelParams())
    , visualizationMode(VisualizationMode::SLICE_XY)
    , sampleRate(44100)
    , audioEnabled(false) {
    
    std::cout << "AirCavityComponent '" << name << "' created" << std::endl;
}

AirCavityComponent::~AirCavityComponent() {
    std::cout << "AirCavityComponent '" << name << "' destroyed" << std::endl;
    
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

void AirCavityComponent::initialize() {
    // Get parameters from simulation manager
    const auto& params = simulationManager.getParameters();
    cellSize = params.cellSize;
    
    // Use full grid dimensions for the air cavity
    cavityWidth = params.gridSizeX;
    cavityHeight = params.gridSizeY;
    cavityDepth = params.gridSizeZ;
    
    std::cout << "Air cavity dimensions: " << cavityWidth << "x" << cavityHeight << "x" << cavityDepth
              << " (cell size: " << cellSize << ")" << std::endl;
    
    // Request grid region from SimulationManager
    // We want to use the entire available grid for the air cavity
    region = simulationManager.findAndAllocateGridRegion(
        std::dynamic_pointer_cast<ComponentInterface>(shared_from_this()),
        cavityWidth, cavityHeight, cavityDepth
    );
    
    if (region.sizeX == 0 || region.sizeY == 0 || region.sizeZ == 0) {
        throw std::runtime_error("Failed to allocate grid region for air cavity");
    }
    
    std::cout << "Allocated grid region at (" << region.startX << "," << region.startY << "," 
              << region.startZ << ") of size " << region.sizeX << "x" << region.sizeY 
              << "x" << region.sizeZ << std::endl;
    
    // Allocate CUDA device buffers
    int pressureElements = cavityWidth * cavityHeight * cavityDepth;
    d_pressure = memoryManager.allocateBuffer<float>(pressureElements);
    d_prevPressure = memoryManager.allocateBuffer<float>(pressureElements);
    
    // Allocate velocity fields using staggered grid approach
    // X velocity has one more element in X direction
    int velocityXElements = (cavityWidth + 1) * cavityHeight * cavityDepth;
    d_velocityX = memoryManager.allocateBuffer<float>(velocityXElements);
    
    // Y velocity has one more element in Y direction
    int velocityYElements = cavityWidth * (cavityHeight + 1) * cavityDepth;
    d_velocityY = memoryManager.allocateBuffer<float>(velocityYElements);
    
    // Z velocity has one more element in Z direction
    int velocityZElements = cavityWidth * cavityHeight * (cavityDepth + 1);
    d_velocityZ = memoryManager.allocateBuffer<float>(velocityZElements);
    
    // Allocate host buffer for data retrieval when needed
    h_pressure.resize(pressureElements, 0.0f);
    
    // Set up kernel parameters
    kernelParams->cavityWidth = cavityWidth;
    kernelParams->cavityHeight = cavityHeight;
    kernelParams->cavityDepth = cavityDepth;
    kernelParams->speedOfSound = speedOfSound;
    kernelParams->density = density;
    kernelParams->dampingCoefficient = dampingCoefficient;
    kernelParams->cellSize = cellSize;
    kernelParams->courantNumber = 0.5f; // Conservative value for stability
    
    // Initialize fields to zero
    ::drumforge::initializeFields(
        d_pressure->get(), d_prevPressure->get(),
        d_velocityX->get(), d_velocityY->get(), d_velocityZ->get(),
        *kernelParams
    );
    
    // Set initial visualization slice to middle of Z dimension
    currentSliceZ = cavityDepth / 2;
    
    // Set up audio buffer
    if (audioEnabled) {
        // Size depends on expected update rate and desired latency
        audioBuffer.resize(sampleRate / 10); // 100ms buffer at 44.1kHz
    }
    
    std::cout << "AirCavityComponent '" << name << "' initialized successfully" << std::endl;
    std::cout << "  Speed of sound: " << speedOfSound << " m/s" << std::endl;
    std::cout << "  Air density: " << density << " kg/m³" << std::endl;
    std::cout << "  Damping coefficient: " << dampingCoefficient << std::endl;
    std::cout << "  Stable timestep: " << calculateStableTimestep() << " s" << std::endl;
}

void AirCavityComponent::update(float timestep) {
    // Calculate a stable timestep if the provided one is too large
    float stableTimestep = calculateStableTimestep();
    float safeTimestep = (timestep > stableTimestep) ? stableTimestep : timestep;
    
    // 1. Update velocity fields based on pressure gradients
    ::drumforge::updateVelocity(
        d_pressure->get(),
        d_velocityX->get(), d_velocityY->get(), d_velocityZ->get(),
        *kernelParams, safeTimestep
    );
    
    // 2. Apply boundary conditions to velocity fields
    ::drumforge::applyBoundaryConditions(
        d_pressure->get(),
        d_velocityX->get(), d_velocityY->get(), d_velocityZ->get(),
        *kernelParams
    );
    
    // 3. Update pressure field based on velocity divergence
    ::drumforge::updatePressure(
        d_pressure->get(), d_prevPressure->get(),
        d_velocityX->get(), d_velocityY->get(), d_velocityZ->get(),
        *kernelParams, safeTimestep
    );
    
    // 4. Apply boundary conditions to pressure field
    ::drumforge::applyBoundaryConditions(
        d_pressure->get(),
        d_velocityX->get(), d_velocityY->get(), d_velocityZ->get(),
        *kernelParams
    );
    
    // 5. Update microphone samples if enabled
    if (audioEnabled && !microphones.empty()) {
        updateMicrophoneSamples(safeTimestep);
    }
}

void AirCavityComponent::prepareForVisualization() {
    // Update CPU-side pressure field for visualization
    d_pressure->copyToHost(h_pressure.data());
    
    // The visualization data update will be handled in the visualize method
    // based on the current visualization mode
}

CouplingData AirCavityComponent::getInterfaceData() {
    // For now, return an empty struct. This will be expanded when we implement coupling.
    return CouplingData();
}

void AirCavityComponent::setCouplingData(const CouplingData& data) {
    // To be implemented when coupling with membrane component
}

std::string AirCavityComponent::getName() const {
    return name;
}

float AirCavityComponent::calculateStableTimestep() const {
    // Use the helper function from the CUDA kernels
    return ::drumforge::calculateStableTimestep(*kernelParams);
}

//-----------------------------------------------------------------------------
// Visualization Methods
//-----------------------------------------------------------------------------

void AirCavityComponent::initializeVisualization(VisualizationManager& visManager) {
    // Check if already initialized
    if (vaoId != 0) {
        return;
    }
    
    std::cout << "Initializing visualization for AirCavityComponent '" << name << "'..." << std::endl;
    
    try {
        // The visualization depends on the selected mode
        // For now, we'll implement XY slice visualization (top view of the cavity)
        
        // Determine the grid size based on visualization mode
        int gridWidth = 0;
        int gridHeight = 0;
        
        switch (visualizationMode) {
            case VisualizationMode::SLICE_XY:
                gridWidth = cavityWidth;
                gridHeight = cavityHeight;
                break;
                
            case VisualizationMode::SLICE_XZ:
                gridWidth = cavityWidth;
                gridHeight = cavityDepth;
                break;
                
            case VisualizationMode::SLICE_YZ:
                gridWidth = cavityHeight;
                gridHeight = cavityDepth;
                break;
                
            case VisualizationMode::VOLUME:
                // For volume visualization, we would use a different approach
                // For now, default to XY slice
                gridWidth = cavityWidth;
                gridHeight = cavityHeight;
                visualizationMode = VisualizationMode::SLICE_XY;
                break;
        }
        
        // Initialize vertices for the grid
        std::vector<float> vertices(gridWidth * gridHeight * 3);
        for (int y = 0; y < gridHeight; y++) {
            for (int x = 0; x < gridWidth; x++) {
                int index = (y * gridWidth + x) * 3;
                
                // Calculate position in world space
                float xPos = (x - gridWidth / 2.0f) * cellSize;
                float yPos = (y - gridHeight / 2.0f) * cellSize;
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
        
        // Generate indices for wireframe rendering
        std::vector<unsigned int> indices;
        
        // Generate row lines
        for (int y = 0; y < gridHeight; y++) {
            for (int x = 0; x < gridWidth - 1; x++) {
                indices.push_back(y * gridWidth + x);
                indices.push_back(y * gridWidth + x + 1);
            }
        }
        
        // Generate column lines
        for (int x = 0; x < gridWidth; x++) {
            for (int y = 0; y < gridHeight - 1; y++) {
                indices.push_back(y * gridWidth + x);
                indices.push_back((y + 1) * gridWidth + x);
            }
        }
        
        // Create Element Buffer Object
        eboId = visManager.createIndexBuffer(indices.data(), indices.size() * sizeof(unsigned int));
        
        std::cout << "Visualization for AirCavityComponent '" << name << "' initialized successfully" << std::endl;
    }
    catch (const std::exception& e) {
        std::cerr << "Error initializing visualization: " << e.what() << std::endl;
        // Don't rethrow - we want to continue even if visualization fails
    }
}

void AirCavityComponent::visualize(VisualizationManager& visManager) {
    // Update visualization data based on current mode
    updateVisualizationData();
    
    // Tell the visualization manager to render this component
    // The air cavity uses a wireframe with blue/red color for pressure
    visManager.renderWireframe(
        vaoId,                     // VAO containing the cavity geometry
        eboId,                     // EBO containing the wireframe indices
        getIndexCount(),           // Number of indices to draw
        glm::vec3(0.0f, 0.4f, 0.8f)  // Base color for the cavity
    );
}

void AirCavityComponent::updateVisualizationData() {
    // This method updates OpenGL vertex positions based on pressure values
    // For now, we're just implementing basic visualization without CUDA-GL interop
    
    // The implementation depends on the selected visualization mode
    int gridWidth = 0;
    int gridHeight = 0;
    std::vector<float> vertices;
    
    switch (visualizationMode) {
        case VisualizationMode::SLICE_XY: {
            gridWidth = cavityWidth;
            gridHeight = cavityHeight;
            vertices.resize(gridWidth * gridHeight * 3);
            
            // Ensure currentSliceZ is within bounds
            currentSliceZ = std::min(std::max(0, currentSliceZ), cavityDepth - 1);
            
            for (int y = 0; y < gridHeight; y++) {
                for (int x = 0; x < gridWidth; x++) {
                    int vertexIdx = (y * gridWidth + x) * 3;
                    int pressureIdx = currentSliceZ * (cavityWidth * cavityHeight) + y * cavityWidth + x;
                    
                    // Calculate position
                    float xPos = (x - gridWidth / 2.0f) * cellSize;
                    float yPos = (y - gridHeight / 2.0f) * cellSize;
                    
                    // Use pressure to determine height (z-coordinate)
                    float pressureValue = h_pressure[pressureIdx];
                    float scaleFactor = 10.0f; // Increase for better visibility
                    float zPos = pressureValue * scaleFactor;
                    
                    vertices[vertexIdx] = xPos;
                    vertices[vertexIdx + 1] = yPos;
                    vertices[vertexIdx + 2] = zPos;
                }
            }
            break;
        }
        
        case VisualizationMode::SLICE_XZ: {
            gridWidth = cavityWidth;
            gridHeight = cavityDepth;
            vertices.resize(gridWidth * gridHeight * 3);
            
            // Ensure currentSliceZ (which is actually Y in this case) is within bounds
            int currentSliceY = std::min(std::max(0, currentSliceZ), cavityHeight - 1);
            
            for (int z = 0; z < cavityDepth; z++) {
                for (int x = 0; x < cavityWidth; x++) {
                    int vertexIdx = (z * gridWidth + x) * 3;
                    int pressureIdx = z * (cavityWidth * cavityHeight) + currentSliceY * cavityWidth + x;
                    
                    // Calculate position
                    float xPos = (x - gridWidth / 2.0f) * cellSize;
                    float zPos = (z - gridHeight / 2.0f) * cellSize;
                    
                    // Use pressure to determine height (y-coordinate)
                    float pressureValue = h_pressure[pressureIdx];
                    float scaleFactor = 10.0f;
                    float yPos = pressureValue * scaleFactor;
                    
                    vertices[vertexIdx] = xPos;
                    vertices[vertexIdx + 1] = yPos;
                    vertices[vertexIdx + 2] = zPos;
                }
            }
            break;
        }
        
        case VisualizationMode::SLICE_YZ: {
            gridWidth = cavityHeight;
            gridHeight = cavityDepth;
            vertices.resize(gridWidth * gridHeight * 3);
            
            // Ensure currentSliceZ (which is actually X in this case) is within bounds
            int currentSliceX = std::min(std::max(0, currentSliceZ), cavityWidth - 1);
            
            for (int z = 0; z < cavityDepth; z++) {
                for (int y = 0; y < cavityHeight; y++) {
                    int vertexIdx = (z * gridWidth + y) * 3;
                    int pressureIdx = z * (cavityWidth * cavityHeight) + y * cavityWidth + currentSliceX;
                    
                    // Calculate position
                    float yPos = (y - gridWidth / 2.0f) * cellSize;
                    float zPos = (z - gridHeight / 2.0f) * cellSize;
                    
                    // Use pressure to determine height (x-coordinate)
                    float pressureValue = h_pressure[pressureIdx];
                    float scaleFactor = 10.0f;
                    float xPos = pressureValue * scaleFactor;
                    
                    vertices[vertexIdx] = xPos;
                    vertices[vertexIdx + 1] = yPos;
                    vertices[vertexIdx + 2] = zPos;
                }
            }
            break;
        }
        
        case VisualizationMode::VOLUME:
            // 3D volume visualization is more complex and not implemented yet
            // For now, fallback to XY slice
            visualizationMode = VisualizationMode::SLICE_XY;
            updateVisualizationData();
            return;
    }
    
    // Update the OpenGL buffer with the new vertex data
    if (vaoId > 0 && !vertices.empty()) {
        GLuint vbo;
        glBindVertexArray(vaoId);
        glGetVertexAttribIuiv(0, GL_VERTEX_ATTRIB_ARRAY_BUFFER_BINDING, &vbo);
        glBindBuffer(GL_ARRAY_BUFFER, vbo);
        glBufferSubData(GL_ARRAY_BUFFER, 0, vertices.size() * sizeof(float), vertices.data());
        glBindBuffer(GL_ARRAY_BUFFER, 0);
        glBindVertexArray(0);
    }
}

int AirCavityComponent::getVertexCount() const {
    // Count depends on visualization mode
    switch (visualizationMode) {
        case VisualizationMode::SLICE_XY:
            return cavityWidth * cavityHeight;
        case VisualizationMode::SLICE_XZ:
            return cavityWidth * cavityDepth;
        case VisualizationMode::SLICE_YZ:
            return cavityHeight * cavityDepth;
        case VisualizationMode::VOLUME:
            // Not fully implemented yet
            return cavityWidth * cavityHeight;
    }
    
    return 0;
}

int AirCavityComponent::getIndexCount() const {
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
// AirCavity-specific Methods
//-----------------------------------------------------------------------------

void AirCavityComponent::reset() {
    // Reset the simulation to initial state
    ::drumforge::initializeFields(
        d_pressure->get(), d_prevPressure->get(),
        d_velocityX->get(), d_velocityY->get(), d_velocityZ->get(),
        *kernelParams
    );
    
    // Clear microphone buffers
    for (auto& mic : microphones) {
        mic.samples.clear();
    }
    
    std::cout << "Air cavity reset to initial state" << std::endl;
}

void AirCavityComponent::addPressureImpulse(float x, float y, float z, float strength, float radius) {
    // Add a pressure impulse at the specified position
    ::drumforge::addPressureImpulse(
        d_pressure->get(),
        *kernelParams,
        x, y, z,
        strength, radius
    );
    
    std::cout << "Added pressure impulse at position (" << x << "," << y << "," << z
              << ") with strength " << strength << " and radius " << radius << std::endl;
}

int AirCavityComponent::addMicrophone(float x, float y, float z, int sampleRate, float gain) {
    // Convert world coordinates to grid coordinates
    int gridX, gridY, gridZ;
    worldToGrid(x, y, z, gridX, gridY, gridZ);
    
    // Check if position is within grid bounds
    if (gridX < 0 || gridX >= cavityWidth || 
        gridY < 0 || gridY >= cavityHeight || 
        gridZ < 0 || gridZ >= cavityDepth) {
        std::cerr << "Warning: Microphone position outside grid boundaries. Clamping to nearest valid position." << std::endl;
        
        gridX = std::min(std::max(0, gridX), cavityWidth - 1);
        gridY = std::min(std::max(0, gridY), cavityHeight - 1);
        gridZ = std::min(std::max(0, gridZ), cavityDepth - 1);
    }
    
    // Create a new microphone
    VirtualMicrophone mic;
    mic.position = glm::vec3(x, y, z);
    mic.gridX = gridX;
    mic.gridY = gridY;
    mic.gridZ = gridZ;
    mic.gain = gain;
    mic.sampleRate = sampleRate;
    
    // Add to microphones list
    microphones.push_back(mic);
    
    // Return the index of the new microphone
    return static_cast<int>(microphones.size() - 1);
}

void AirCavityComponent::removeMicrophone(int index) {
    if (index >= 0 && index < static_cast<int>(microphones.size())) {
        microphones.erase(microphones.begin() + index);
    } else {
        std::cerr << "Warning: Invalid microphone index in removeMicrophone" << std::endl;
    }
}

const std::vector<float>& AirCavityComponent::getMicrophoneSamples(int index) const {
    static const std::vector<float> emptyVector;
    
    if (index >= 0 && index < static_cast<int>(microphones.size())) {
        return microphones[index].samples;
    }
    
    return emptyVector;
}

void AirCavityComponent::updateMicrophoneSamples(float timestep) {
    // Update CPU-side pressure field if needed
    if (h_pressure.empty() || h_pressure.size() != cavityWidth * cavityHeight * cavityDepth) {
        h_pressure.resize(cavityWidth * cavityHeight * cavityDepth);
        d_pressure->copyToHost(h_pressure.data());
    }
    
    // For each microphone, sample the pressure field
    for (auto& mic : microphones) {
        // Calculate pressure at microphone position
        int pressureIdx = mic.gridZ * (cavityWidth * cavityHeight) + 
                         mic.gridY * cavityWidth + 
                         mic.gridX;
        
        // Get pressure value
        float pressure = h_pressure[pressureIdx];
        
        // Apply gain
        pressure *= mic.gain;
        
        // Add to microphone's sample buffer
        mic.samples.push_back(pressure);
        
        // Limit buffer size (to avoid excessive memory usage)
        const size_t maxSamples = 44100 * 5; // 5 seconds at 44.1kHz
        if (mic.samples.size() > maxSamples) {
            mic.samples.erase(mic.samples.begin(), mic.samples.begin() + 
                             (mic.samples.size() - maxSamples));
        }
    }
}

void AirCavityComponent::enableAudio(bool enable) {
    audioEnabled = enable;
    
    // If enabling audio, resize audio buffer if needed
    if (audioEnabled) {
        audioBuffer.resize(sampleRate / 10); // 100ms buffer
    }
}

void AirCavityComponent::setSampleRate(int rate) {
    if (rate > 0) {
        sampleRate = rate;
        
        // Update microphones to match new sample rate
        for (auto& mic : microphones) {
            mic.sampleRate = rate;
        }
        
        // Resize audio buffer if needed
        if (audioEnabled) {
            audioBuffer.resize(sampleRate / 10); // 100ms buffer
        }
    }
}

void AirCavityComponent::setSpeedOfSound(float speed) {
    if (speed > 0.0f) {
        speedOfSound = speed;
        kernelParams->speedOfSound = speed;
        
        std::cout << "Air cavity speed of sound updated to " << speed << " m/s" << std::endl;
    } else {
        std::cerr << "Warning: Invalid speed of sound value " << speed << std::endl;
    }
}

void AirCavityComponent::setDensity(float density) {
    if (density > 0.0f) {
        this->density = density;
        kernelParams->density = density;
        
        std::cout << "Air cavity density updated to " << density << " kg/m³" << std::endl;
    } else {
        std::cerr << "Warning: Invalid density value " << density << std::endl;
    }
}

void AirCavityComponent::setDampingCoefficient(float damping) {
    if (damping >= 0.0f) {
        dampingCoefficient = damping;
        kernelParams->dampingCoefficient = damping;
        
        std::cout << "Air cavity damping coefficient updated to " << damping << std::endl;
    } else {
        std::cerr << "Warning: Invalid damping coefficient value " << damping << std::endl;
    }
}

void AirCavityComponent::setVisualizationMode(VisualizationMode mode) {
    if (mode != visualizationMode) {
        visualizationMode = mode;
        
        // Recreate visualization resources if needed
        if (vaoId != 0) {
            // In a real implementation, we would recreate the VAO and EBO
            // For simplicity, we're just setting it to 0 to force recreation
            vaoId = 0;
            eboId = 0;
            
            // Recreate visualization in next frame
            // We'd need VisualizationManager reference to do this properly
        }
    }
}

void AirCavityComponent::setCurrentSliceZ(int z) {
    // Clamp to valid range based on visualization mode
    switch (visualizationMode) {
        case VisualizationMode::SLICE_XY:
            currentSliceZ = std::min(std::max(0, z), cavityDepth - 1);
            break;
        case VisualizationMode::SLICE_XZ:
            currentSliceZ = std::min(std::max(0, z), cavityHeight - 1);
            break;
        case VisualizationMode::SLICE_YZ:
            currentSliceZ = std::min(std::max(0, z), cavityWidth - 1);
            break;
        case VisualizationMode::VOLUME:
            // Not applicable for volume visualization
            break;
    }
}

float AirCavityComponent::getPressureAt(float x, float y, float z) const {
    // Convert world coordinates to grid coordinates
    int gridX, gridY, gridZ;
    worldToGrid(x, y, z, gridX, gridY, gridZ);
    
    return getPressureAtGrid(gridX, gridY, gridZ);
}

float AirCavityComponent::getPressureAtGrid(int x, int y, int z) const {
    // Check if coordinates are within valid range
    if (x < 0 || x >= cavityWidth || 
        y < 0 || y >= cavityHeight || 
        z < 0 || z >= cavityDepth) {
        return 0.0f;
    }
    
    // Update CPU-side pressure field if needed
    if (h_pressure.empty() || h_pressure.size() != cavityWidth * cavityHeight * cavityDepth) {
        const_cast<AirCavityComponent*>(this)->h_pressure.resize(cavityWidth * cavityHeight * cavityDepth);
        const_cast<AirCavityComponent*>(this)->d_pressure->copyToHost(h_pressure.data());
    }
    
    // Calculate index in the 3D grid
    int index = z * (cavityWidth * cavityHeight) + y * cavityWidth + x;
    
    // Return pressure value
    return h_pressure[index];
}

const std::vector<float>& AirCavityComponent::getPressureField() const {
    // Update CPU-side pressure field
    const_cast<AirCavityComponent*>(this)->d_pressure->copyToHost(
        const_cast<AirCavityComponent*>(this)->h_pressure.data()
    );
    
    return h_pressure;
}

glm::vec3 AirCavityComponent::gridToWorld(int x, int y, int z) const {
    // Convert from local grid coordinates to global world coordinates
    int globalX, globalY, globalZ;
    simulationManager.localToGlobal(region, x, y, z, globalX, globalY, globalZ);
    
    // Convert global grid coordinates to world coordinates
    return simulationManager.gridToWorld(globalX, globalY, globalZ);
}

void AirCavityComponent::worldToGrid(float x, float y, float z, int& outX, int& outY, int& outZ) const {
    // Convert from world coordinates to global grid coordinates
    int globalX, globalY, globalZ;
    simulationManager.worldToGrid(x, y, z, globalX, globalY, globalZ);
    
    // Convert from global grid coordinates to local coordinates
    simulationManager.globalToLocal(region, globalX, globalY, globalZ, outX, outY, outZ);
}

void AirCavityComponent::applyBoundaryConditions() {
    // This method is called internally by update()
    // The actual implementation happens in the CUDA kernel
    // This is just a stub in case we need to add component-level logic
}

} // namespace drumforge