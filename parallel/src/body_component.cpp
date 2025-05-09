#include "body_component.h"
#include "body_kernels.cuh"
#include "visualization_manager.h"
#include <stdexcept>
#include <iostream>
#include <cmath>
#include <vector>
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace drumforge {

//-----------------------------------------------------------------------------
// Constructor & Destructor
//-----------------------------------------------------------------------------

BodyComponent::BodyComponent(
    const std::string& name,
    float radius,
    float height,
    float thickness,
    const std::string& material)
    : memoryManager(CudaMemoryManager::getInstance())
    , simulationManager(SimulationManager::getInstance())
    , radius(radius)
    , height(height)
    , thickness(thickness)
    , masterGain(1.0f)
    , useMixedOutput(true)
    , material(material)
    , cellSize(0.0f)
    , vaoId(0)
    , eboId(0)
    , name(name)
    , kernelParams(new BodyKernelParams()) {
    
    std::cout << "BodyComponent '" << name << "' created" << std::endl;
    std::cout << "  Dimensions: radius=" << radius 
              << ", height=" << height
              << ", thickness=" << thickness << std::endl;
    std::cout << "  Material: " << material << std::endl;
    
    // Setup default microphones
    setupDefaultMicrophones();
}

BodyComponent::~BodyComponent() {
    std::cout << "BodyComponent '" << name << "' destroyed" << std::endl;
    
    // Release grid region if allocated
    if (region.sizeX > 0 && region.sizeY > 0 && region.sizeZ > 0) {
        simulationManager.releaseGridRegion(region);
    }
    
    // OpenGL resources will be cleaned up by the VisualizationManager
    
    // CUDA buffers are automatically released by shared_ptr destruction
}

//-----------------------------------------------------------------------------
// ComponentInterface Implementation
//-----------------------------------------------------------------------------

void BodyComponent::initialize() {
    // Get parameters from simulation manager
    const auto& params = simulationManager.getParameters();
    cellSize = params.cellSize;
    
    // Determine body dimensions based on radius and height
    // Convert from physical dimensions to grid cells
    int gridRadius = static_cast<int>(radius / cellSize);
    int gridHeight = static_cast<int>(height / cellSize);
    
    // Add padding for visualization
    const int padding = 2;
    int gridDiameter = gridRadius * 2 + 2 * padding;
    
    // Ensure grid dimensions are even for symmetry
    if (gridDiameter % 2 != 0) {
        gridDiameter++;
    }
    if (gridHeight % 2 != 0) {
        gridHeight++;
    }
    
    std::cout << "Body grid dimensions: " << gridDiameter << "x" << gridDiameter 
              << "x" << gridHeight << " (radius: " << radius 
              << ", cell size: " << cellSize << ")" << std::endl;
    
    // Request grid region from SimulationManager for visualization purposes
    // The body itself doesn't use the grid for simulation, but we allocate it
    // to ensure proper positioning with other components
    region = simulationManager.findAndAllocateGridRegion(
        std::dynamic_pointer_cast<ComponentInterface>(shared_from_this()),
        gridDiameter, gridDiameter, gridHeight
    );
    
    if (region.sizeX == 0 || region.sizeY == 0 || region.sizeZ == 0) {
        throw std::runtime_error("Failed to allocate grid region for body");
    }
    
    std::cout << "Allocated grid region at (" << region.startX << "," << region.startY << "," 
              << region.startZ << ") of size " << region.sizeX << "x" << region.sizeY 
              << "x" << region.sizeZ << std::endl;
    
    // Set up kernel parameters
    kernelParams->radius = radius;
    kernelParams->height = height;
    kernelParams->thickness = thickness;
    kernelParams->cellSize = cellSize;
    
    // Set up material preset
    setupMaterialPreset(material);
    
    // Initialize modes
    setupDefaultModes();
    
    // Allocate CUDA device buffers
    int numModes = kernelParams->numModes;
    d_modeStates = memoryManager.allocateBuffer<float>(numModes);
    d_modeVelocities = memoryManager.allocateBuffer<float>(numModes);
    d_modes = memoryManager.allocateBuffer<ResonantMode>(numModes);
    d_excitation = memoryManager.allocateBuffer<float>(numModes);
    
    // Zero out the buffers
    d_modeStates->zero();
    d_modeVelocities->zero();
    d_excitation->zero();
    
    // Allocate host buffers for data retrieval
    h_modeStates.resize(numModes, 0.0f);
    h_modeVelocities.resize(numModes, 0.0f);
    
    // Initialize the modes on the GPU
    initializeModes(d_modes->get(), *kernelParams);
    
    // Reset the body to its initial state
    resetBody(d_modeStates->get(), d_modeVelocities->get(), numModes);
    
    std::cout << "BodyComponent '" << name << "' initialized with " 
              << numModes << " modes" << std::endl;
}

void BodyComponent::update(float timestep) {
    // Calculate a stable timestep if the provided one is too large
    float stableTimestep = calculateStableTimestep();
    float safeTimestep = (timestep > stableTimestep) ? stableTimestep : timestep;
    
    // Update the modal synthesis simulation
    updateBodyModes(
        d_modeStates->get(),
        d_modeVelocities->get(),
        d_modes->get(),
        d_excitation->get(),
        kernelParams->numModes,
        safeTimestep
    );
    
    // Clear excitation after update (excitation is an impulse)
    // This ensures that excitation is applied only for one step
    d_excitation->zero();
    
    // Update audio if needed
    updateAudio(timestep);
}

void BodyComponent::prepareForVisualization() {
    // For the body visualization, we don't need to update anything
    // The visualization is based on a static cylinder representing the shell
    // No CUDA-OpenGL interop is needed here
}

CouplingData BodyComponent::getInterfaceData() {
    // For now, return an empty struct.
    // This will be expanded when we implement coupling with membranes.
    return CouplingData();
}

void BodyComponent::setCouplingData(const CouplingData& data) {
    // To be implemented for coupling with the membrane:
    // 1. Extract membrane displacements from coupling data
    // 2. Calculate modal excitation from membrane displacements
    // 3. Apply excitation to the modes
}

std::string BodyComponent::getName() const {
    return name;
}

float BodyComponent::calculateStableTimestep() const {
    // Use the kernel helper to calculate a stable timestep
    return ::drumforge::calculateStableTimestep(*kernelParams);
}

//-----------------------------------------------------------------------------
// Visualization Methods
//-----------------------------------------------------------------------------

void BodyComponent::initializeVisualization(VisualizationManager& visManager) {
    // Check if already initialized
    if (vaoId != 0) {
        return;
    }
    
    std::cout << "Initializing visualization for BodyComponent '" << name << "'..." << std::endl;
    
    try {
        // Create vertices for a cylindrical shell
        std::vector<float> vertices;
        std::vector<unsigned int> indices;
        
        // Parameters for the cylinder
        const int numSegments = 36;  // Number of segments around the circumference
        const int numRings = 8;      // Number of rings along the height
        
        // Calculate vertices
        for (int ring = 0; ring <= numRings; ring++) {
            // Use Z for height (instead of Y as before)
            // We want the drum membrane to be at Z=0, so the body extends into negative Z
            float z = -((static_cast<float>(ring) / numRings) * height);
            
            for (int segment = 0; segment <= numSegments; segment++) {
                float angle = (static_cast<float>(segment) / numSegments) * 2.0f * M_PI;
                
                // Use X and Y for the circular cross-section (instead of X and Z as before)
                float x = radius * cos(angle);
                float y = radius * sin(angle);
                
                // Add vertex position (x, y, z) - note the changed order
                vertices.push_back(x);
                vertices.push_back(y);
                vertices.push_back(z);
            }
        }
        
        // Calculate indices for wireframe rendering
        for (int ring = 0; ring < numRings; ring++) {
            for (int segment = 0; segment < numSegments; segment++) {
                // Get indices of the current quad's corners
                unsigned int topLeft = ring * (numSegments + 1) + segment;
                unsigned int topRight = topLeft + 1;
                unsigned int bottomLeft = (ring + 1) * (numSegments + 1) + segment;
                unsigned int bottomRight = bottomLeft + 1;
                
                // Add lines for the wireframe
                // Horizontal lines (around circumference)
                indices.push_back(topLeft);
                indices.push_back(topRight);
                
                // Vertical lines (along height)
                indices.push_back(topLeft);
                indices.push_back(bottomLeft);
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
        
        // Create Element Buffer Object for wireframe indices
        eboId = visManager.createIndexBuffer(indices.data(), indices.size() * sizeof(unsigned int));
        
        std::cout << "Visualization for BodyComponent '" << name << "' initialized successfully" << std::endl;
    }
    catch (const std::exception& e) {
        std::cerr << "Error initializing visualization: " << e.what() << std::endl;
        // Don't rethrow - we want to continue even if visualization fails
    }
}

void BodyComponent::visualize(VisualizationManager& visManager) {
    // The body now uses the same coordinate system as the membrane,
    // so we don't need any special transformation
    visManager.renderWireframe(
        vaoId,                      // VAO containing the body geometry
        eboId,                      // EBO containing the wireframe indices
        getNumWireframeIndices(),   // Number of indices to draw
        glm::vec3(0.7f, 0.4f, 0.2f) // Brownish color for wood
    );
}

int BodyComponent::getNumWireframeIndices() const {
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
// Audio Methods
//-----------------------------------------------------------------------------

void BodyComponent::initializeAudioChannels() {
    // Get reference to AudioManager
    AudioManager& audioManager = AudioManager::getInstance();
    
    // Clear any existing channels
    for (int channelIdx : audioChannelIndices) {
        audioManager.removeChannel(channelIdx);
    }
    audioChannelIndices.clear();
    
    // Register a channel for each microphone
    for (int i = 0; i < static_cast<int>(microphones.size()); ++i) {
        const auto& mic = microphones[i];
        int channelIdx = audioManager.addChannel(
            name + " - " + mic.name,  // Use component name + mic name
            mic.gain                  // Use mic gain as initial channel gain
        );
        audioChannelIndices.push_back(channelIdx);
    }
    
    audioChannelsInitialized = true;
}

void BodyComponent::updateAudio(float timestep) {
    // Get reference to AudioManager
    AudioManager& audioManager = AudioManager::getInstance();
    
    // Only sample if recording is active
    if (!audioManager.getIsRecording()) {
        return;
    }
    
    // Initialize audio channels if needed
    if (!audioChannelsInitialized || audioChannelIndices.empty()) {
        initializeAudioChannels();
    }
    
    // Copy current mode states from device to host
    getModeStates();
    
    // Get the modes from device if needed for microphone sampling
    std::vector<ResonantMode> modes(kernelParams->numModes);
    d_modes->copyToHost(modes.data());
    
    // Process for each microphone
    if (useMixedOutput) {
        // Approach 1: Collect signals from all microphones and mix them
        float mixedSignal = 0.0f;
        int activeMicCount = 0;
        
        for (int i = 0; i < static_cast<int>(microphones.size()); ++i) {
            const auto& mic = microphones[i];
            if (!mic.enabled) continue;
            
            // Sample this microphone position
            float sample = 0.0f;
            
            // Calculate sample by summing modal contributions with position weighting
            for (int modeIdx = 0; modeIdx < kernelParams->numModes; modeIdx++) {
                float amplitude = modes[modeIdx].amplitude;
                float state = h_modeStates[modeIdx];
                
                // Calculate spatial weighting based on microphone position
                float modeNumber = static_cast<float>(modeIdx + 1);
                float theta = mic.position.x * 2.0f * M_PI;
                float heightPos = mic.height;
                
                // Spatial weighting (similar to kernel function)
                float spatialWeight = cos(modeNumber * theta) * 
                                     sin(modeNumber * heightPos * M_PI);
                
                // Add contribution
                sample += amplitude * state * spatialWeight;
            }
            
            // Apply microphone gain
            sample *= mic.gain;
            
            // Update channel value for monitoring
            if (i < static_cast<int>(audioChannelIndices.size())) {
                audioManager.setChannelValue(audioChannelIndices[i], sample);
            }
            
            // Add to mix
            mixedSignal += sample;
            activeMicCount++;
        }
        
        // Apply mixing formula and master gain
        if (activeMicCount > 0) {
            mixedSignal /= sqrt(static_cast<float>(activeMicCount));
            mixedSignal *= masterGain;
            
            // Process the mixed sample
            audioManager.processAudioStep(timestep, mixedSignal);
        }
    } 
    else {
        // Approach 2: Only process the first active microphone
        for (int i = 0; i < static_cast<int>(microphones.size()); ++i) {
            const auto& mic = microphones[i];
            if (!mic.enabled) continue;
            
            // Sample this microphone position
            float sample = 0.0f;
            
            // Calculate sample by summing modal contributions with position weighting
            for (int modeIdx = 0; modeIdx < kernelParams->numModes; modeIdx++) {
                float amplitude = modes[modeIdx].amplitude;
                float state = h_modeStates[modeIdx];
                
                // Calculate spatial weighting based on microphone position
                float modeNumber = static_cast<float>(modeIdx + 1);
                float theta = mic.position.x * 2.0f * M_PI;
                float heightPos = mic.height;
                
                // Spatial weighting (similar to kernel function)
                float spatialWeight = cos(modeNumber * theta) * 
                                     sin(modeNumber * heightPos * M_PI);
                
                // Add contribution
                sample += amplitude * state * spatialWeight;
            }
            
            // Apply microphone gain and master gain
            sample *= mic.gain * masterGain;
            
            // Update channel value for monitoring
            if (i < static_cast<int>(audioChannelIndices.size())) {
                audioManager.setChannelValue(audioChannelIndices[i], sample);
            }
            
            // Process just this single microphone
            audioManager.processAudioStep(timestep, sample);
            
            // Use only the first active microphone
            break;
        }
    }
}

//-----------------------------------------------------------------------------
// Body-specific Methods
//-----------------------------------------------------------------------------

void BodyComponent::reset() {
    // Reset the body to its initial state
    resetBody(d_modeStates->get(), d_modeVelocities->get(), kernelParams->numModes);
    
    std::cout << "Body reset to initial state" << std::endl;
}

void BodyComponent::exciteMode(int modeIndex, float amount) {
    // Apply excitation to a specific mode or all modes if modeIndex is -1
    ::drumforge::exciteMode(d_excitation->get(), modeIndex, amount, kernelParams->numModes);
    
    if (modeIndex >= 0) {
        std::cout << "Excited body mode " << modeIndex << " with amount " << amount << std::endl;
    } else {
        std::cout << "Excited all body modes with amount " << amount << std::endl;
    }
}

void BodyComponent::exciteAllModes(const std::vector<float>& excitation) {
    // Check if the excitation vector has the correct size
    if (excitation.size() != static_cast<size_t>(kernelParams->numModes)) {
        throw std::invalid_argument("Excitation vector size does not match number of modes");
    }
    
    // Copy excitation to device
    // Create a temporary device buffer for the input excitation
    auto d_inputExcitation = memoryManager.allocateBuffer<float>(excitation.size());
    d_inputExcitation->copyFromHost(excitation.data());
    
    // Apply excitation to all modes
    ::drumforge::exciteAllModes(d_excitation->get(), d_inputExcitation->get(), kernelParams->numModes);
    
    std::cout << "Applied custom excitation to body modes" << std::endl;
}

void BodyComponent::setupDefaultModes() {
    // Use default number of modes
    int numModes = 64;
    kernelParams->numModes = numModes;
    
    // Other parameters are set by the material preset in setupMaterialPreset
    
    std::cout << "Set up " << numModes << " default modes for body resonator" << std::endl;
}

void BodyComponent::setupMaterialPreset(const std::string& material) {
    // Use the kernel helper to set up the material preset
    ::drumforge::setupMaterialPreset(*kernelParams, material);
    
    std::cout << "Applied material preset for '" << material << "'" << std::endl;
}

//-----------------------------------------------------------------------------
// Parameter Setters
//-----------------------------------------------------------------------------

void BodyComponent::setRadius(float newRadius) {
    // Don't allow negative or zero radius
    if (newRadius <= 0.0f) {
        std::cerr << "Warning: Cannot set radius to " << newRadius 
                  << ". Using minimum value of 0.1f instead." << std::endl;
        newRadius = 0.1f;
    }
    
    // Update the component's radius value
    radius = newRadius;
    kernelParams->radius = radius;
    
    std::cout << "Body radius updated to " << radius << std::endl;
    
    // Reinitialize the modes with the new radius
    initializeModes(d_modes->get(), *kernelParams);
    
    // Reset the body with the new parameters
    reset();
}

void BodyComponent::setHeight(float newHeight) {
    // Don't allow negative or zero height
    if (newHeight <= 0.0f) {
        std::cerr << "Warning: Cannot set height to " << newHeight 
                  << ". Using minimum value of 0.1f instead." << std::endl;
        newHeight = 0.1f;
    }
    
    // Update the component's height value
    height = newHeight;
    kernelParams->height = height;
    
    std::cout << "Body height updated to " << height << std::endl;
    
    // Reinitialize the modes with the new height
    initializeModes(d_modes->get(), *kernelParams);
    
    // Reset the body with the new parameters
    reset();
}

void BodyComponent::setThickness(float newThickness) {
    // Don't allow negative or zero thickness
    if (newThickness <= 0.0f) {
        std::cerr << "Warning: Cannot set thickness to " << newThickness 
                  << ". Using minimum value of 0.001f instead." << std::endl;
        newThickness = 0.001f;
    }
    
    // Update the component's thickness value
    thickness = newThickness;
    kernelParams->thickness = thickness;
    
    std::cout << "Body thickness updated to " << thickness << std::endl;
    
    // Reinitialize the modes with the new thickness
    initializeModes(d_modes->get(), *kernelParams);
    
    // Reset the body with the new parameters
    reset();
}

void BodyComponent::setMaterial(const std::string& newMaterial) {
    // Update the component's material value
    material = newMaterial;
    
    // Update material parameters
    setupMaterialPreset(material);
    
    std::cout << "Body material updated to " << material << std::endl;
    
    // Reinitialize the modes with the new material parameters
    initializeModes(d_modes->get(), *kernelParams);
    
    // Reset the body with the new parameters
    reset();
}

//-----------------------------------------------------------------------------
// Microphone Management
//-----------------------------------------------------------------------------

int BodyComponent::addMicrophone(float x, float y, float height, float gain, const std::string& name) {
    // Clamp position to valid range [0,1]
    x = std::max(0.0f, std::min(x, 1.0f));
    y = std::max(0.0f, std::min(y, 1.0f));
    height = std::max(0.0f, std::min(height, 1.0f));
    
    // Create new microphone
    BodyVirtualMicrophone mic;
    mic.position = glm::vec2(x, y);
    mic.height = height;
    mic.gain = gain;
    mic.enabled = true;
    mic.name = name.empty() ? "Mic " + std::to_string(microphones.size() + 1) : name;
    
    // Add to collection
    microphones.push_back(mic);
    
    std::cout << "Added microphone '" << mic.name << "' at position (" 
              << x << ", " << y << ", " << height << ") with gain " << gain << std::endl;
    
    audioChannelsInitialized = false;
    
    return microphones.size() - 1; // Return index of the new microphone
}

void BodyComponent::removeMicrophone(int index) {
    if (index >= 0 && index < static_cast<int>(microphones.size())) {
        std::cout << "Removed microphone '" << microphones[index].name << "'" << std::endl;
        microphones.erase(microphones.begin() + index);
        audioChannelsInitialized = false;
    }
}

void BodyComponent::clearAllMicrophones() {
    microphones.clear();
    std::cout << "All microphones removed" << std::endl;
    audioChannelsInitialized = false;
}

const BodyVirtualMicrophone& BodyComponent::getMicrophone(int index) const {
    static const BodyVirtualMicrophone defaultMic = {
        glm::vec2(0.5f, 0.5f), 0.5f, 1.0f, false, "Invalid"
    };
    
    if (index >= 0 && index < static_cast<int>(microphones.size())) {
        return microphones[index];
    }
    return defaultMic;
}

void BodyComponent::setMicrophonePosition(int index, float x, float y, float height) {
    if (index >= 0 && index < static_cast<int>(microphones.size())) {
        // Clamp position to valid range [0,1]
        x = std::max(0.0f, std::min(x, 1.0f));
        y = std::max(0.0f, std::min(y, 1.0f));
        height = std::max(0.0f, std::min(height, 1.0f));
        
        microphones[index].position = glm::vec2(x, y);
        microphones[index].height = height;
        audioChannelsInitialized = false;
    }
}

void BodyComponent::setMicrophoneGain(int index, float gain) {
    if (index >= 0 && index < static_cast<int>(microphones.size())) {
        microphones[index].gain = gain;
        audioChannelsInitialized = false;
    }
}

void BodyComponent::enableMicrophone(int index, bool enabled) {
    if (index >= 0 && index < static_cast<int>(microphones.size())) {
        microphones[index].enabled = enabled;
        audioChannelsInitialized = false;
    }
}

void BodyComponent::setupDefaultMicrophones() {
    clearAllMicrophones();
    
    // Add 4 microphones around the shell at different heights
    addMicrophone(0.0f, 0.0f, 0.5f, 1.0f, "Front");  // Front center
    addMicrophone(0.5f, 0.0f, 0.7f, 0.8f, "Right");  // Right side, higher
    addMicrophone(1.0f, 0.0f, 0.5f, 0.8f, "Back");   // Back center
    addMicrophone(0.5f, 0.0f, 0.3f, 0.8f, "Left");   // Left side, lower
    
    audioChannelsInitialized = false;
}

//-----------------------------------------------------------------------------
// Data Access Methods
//-----------------------------------------------------------------------------

const std::vector<float>& BodyComponent::getModeStates() const {
    // Copy the current mode states from device to host for CPU access
    // Cast away constness for this operation - the function is logically const
    // even though it updates the cache of mode values
    auto& nonConstThis = const_cast<BodyComponent&>(*this);
    auto& nonConstStates = const_cast<std::vector<float>&>(h_modeStates);
    
    d_modeStates->copyToHost(nonConstStates.data());
    return h_modeStates;
}

const std::vector<float>& BodyComponent::getModeVelocities() const {
    // Copy the current mode velocities from device to host for CPU access
    auto& nonConstThis = const_cast<BodyComponent&>(*this);
    auto& nonConstVelocities = const_cast<std::vector<float>&>(h_modeVelocities);
    
    d_modeVelocities->copyToHost(nonConstVelocities.data());
    return h_modeVelocities;
}

} // namespace drumforge