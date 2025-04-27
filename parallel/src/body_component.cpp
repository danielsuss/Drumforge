#include "body_component.h"
#include "body_kernels.cuh"
#include "visualization_manager.h"
#include <stdexcept>
#include <iostream>
#include <cmath>
#include <vector>
#include <algorithm>
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
}

//-----------------------------------------------------------------------------
// ComponentInterface Implementation
//-----------------------------------------------------------------------------

void BodyComponent::initialize() {
    // Get parameters from simulation manager
    const auto& params = simulationManager.getParameters();
    cellSize = params.cellSize;
    
    // Debug output
    std::cout << "Body initializing with radius=" << radius 
              << ", height=" << height 
              << ", cellSize=" << cellSize 
              << std::endl;
    
    // Determine body dimensions based on radius and height
    // Convert from physical dimensions to grid cells
    int gridRadius = std::max(1, static_cast<int>(radius / cellSize));
    int gridHeight = std::max(1, static_cast<int>(height / cellSize));
    
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
    
    std::cout << "Body grid dimensions calculated: " << gridDiameter << "x" << gridDiameter 
              << "x" << gridHeight << std::endl;
    
    try {
        // IMPORTANT: Use a safer way to convert to shared_ptr
        std::shared_ptr<ComponentInterface> componentPtr = shared_from_this();

        // Request grid region from SimulationManager for visualization purposes
        region = simulationManager.findAndAllocateGridRegion(componentPtr, 
                                                          gridDiameter, 
                                                          gridDiameter, 
                                                          gridHeight);
        
        std::cout << "Allocated grid region at (" << region.startX << "," << region.startY << "," 
                  << region.startZ << ") of size " << region.sizeX << "x" << region.sizeY 
                  << "x" << region.sizeZ << std::endl;
    }
    catch (const std::exception& e) {
        std::cerr << "Exception during grid allocation: " << e.what() << std::endl;
        throw;
    }
    
    // Set up kernel parameters
    kernelParams->radius = radius;
    kernelParams->height = height;
    kernelParams->thickness = thickness;
    
    // Set up material preset
    setupMaterialPreset(material);
    
    // Initialize modes
    setupDefaultModes();
    
    // Create and initialize CUDA buffers
    try {
        int numModes = kernelParams->numModes;
        std::cout << "Allocating CUDA buffers for " << numModes << " modes..." << std::endl;
        
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
        h_modes.resize(numModes);
        
        // Initialize the modes on the GPU
        std::cout << "Initializing modes on GPU..." << std::endl;
        initializeModes(d_modes->get(), *kernelParams);
        
        // Copy modes back to host for debug
        d_modes->copyToHost(h_modes.data());
        
        // Reset the body to its initial state
        resetBody(d_modeStates->get(), d_modeVelocities->get(), numModes);
        
        // Print mode frequencies for debugging
        reportModeFrequencies();
        
        std::cout << "BodyComponent '" << name << "' initialized with " 
                << numModes << " modes" << std::endl;
    }
    catch (const std::exception& e) {
        std::cerr << "Exception during CUDA allocation: " << e.what() << std::endl;
        throw;
    }
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
    d_excitation->zero();
}

void BodyComponent::prepareForVisualization() {
    // For the body visualization, we don't need dynamic updates
    // The visualization is based on a static cylinder shape
}

CouplingData BodyComponent::getInterfaceData() {
    // For now, return an empty struct
    return CouplingData();
}

void BodyComponent::setCouplingData(const CouplingData& data) {
    if (data.hasImpact) {
        std::cout << "Body receiving impact: strength=" << data.impactStrength
                  << ", position=(" << data.impactPosition.x << "," 
                  << data.impactPosition.y << ")" << std::endl;
        
        // Calculate impact position in polar coordinates relative to center
        glm::vec2 normalizedPos = data.impactPosition - glm::vec2(0.5f, 0.5f);
        float radialDist = glm::length(normalizedPos) * 2.0f;  // [0-1] range
        float angle = atan2(normalizedPos.y, normalizedPos.x);  // [-π to π]
        
        // Basic strength scaling
        float strength = data.impactStrength * 10.0f;  // Boost the strength
        
        // Create excitation vector for all modes
        std::vector<float> excitation(kernelParams->numModes, 0.0f);
        
        // Calculate circumferential and axial modes for indexing
        int circumferentialModes = kernelParams->circumferentialModes;
        int axialModes = kernelParams->axialModes;
        
        for (int i = 0; i < kernelParams->numModes; i++) {
            // Calculate mode numbers (n,m)
            int n = (i % circumferentialModes) + 1;  // Circumferential (1,2,3,...)
            int m = (i / circumferentialModes) % axialModes;  // Axial (0,1,2,...)
            
            // Radial position factor - excites modes based on impact distance from center
            float radialFactor = 1.0f - std::pow(std::abs(radialDist - static_cast<float>(n) / circumferentialModes), 2.0f);
            radialFactor = std::max(0.1f, radialFactor);
            
            // Angular position factor - excites modes based on impact angle
            float angularFactor = 0.5f + 0.5f * std::cos(n * angle);
            angularFactor = std::max(0.1f, angularFactor);
            
            // Height position factor - roughly estimated
            float heightFactor = (m == 0) ? 1.0f : 0.5f;  // Reduce excitation for height modes
            
            // Combined excitation for this mode
            excitation[i] = strength * radialFactor * angularFactor * heightFactor;
            
            // Scale based on expected energy distribution - lower modes receive more energy
            excitation[i] /= (1.0f + 0.1f*n + 0.1f*m);
        }
        
        // Apply the calculated excitation to all modes
        exciteAllModes(excitation);
        
        std::cout << "Created excitation vector for " << kernelParams->numModes << " modes" << std::endl;
    }
}

std::string BodyComponent::getName() const {
    return name;
}

float BodyComponent::calculateStableTimestep() const {
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
            // Height position, with membrane at Z=0 and body extending down
            float z = -((static_cast<float>(ring) / numRings) * height);
            
            for (int segment = 0; segment <= numSegments; segment++) {
                float angle = (static_cast<float>(segment) / numSegments) * 2.0f * M_PI;
                
                // X and Y for circular cross-section
                float x = radius * cos(angle);
                float y = radius * sin(angle);
                
                // Add vertex (x, y, z)
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
                
                // Add lines for the wireframe (horizontal and vertical)
                indices.push_back(topLeft);
                indices.push_back(topRight);
                
                indices.push_back(topLeft);
                indices.push_back(bottomLeft);
            }
        }
        
        // Create a vertex buffer
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
    }
}

void BodyComponent::visualize(VisualizationManager& visManager) {
    // Render the body using a wireframe
    visManager.renderWireframe(
        vaoId,                      // VAO containing geometry
        eboId,                      // EBO containing wireframe indices
        getNumWireframeIndices(),   // Number of indices to draw
        glm::vec3(0.7f, 0.4f, 0.2f) // Brownish color for wood
    );
}

int BodyComponent::getNumWireframeIndices() const {
    // If we have no EBO yet, return 0
    if (eboId == 0) {
        return 0;
    }
    
    // Get the count from OpenGL
    GLint count;
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, eboId);
    glGetBufferParameteriv(GL_ELEMENT_ARRAY_BUFFER, GL_BUFFER_SIZE, &count);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
    
    // Convert byte count to index count
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
    // Scale timestep to match membrane's scaling
    timestep *= 1.0f/60.0f;
    
    // Get reference to AudioManager
    AudioManager& audioManager = AudioManager::getInstance();
    
    // Only process if recording is active
    if (!audioManager.getIsRecording()) {
        return;
    }
    
    // Initialize audio channels if needed
    if (!audioChannelsInitialized || audioChannelIndices.empty()) {
        initializeAudioChannels();
    }
    
    // Get mode states from the device
    getModeStates();
    
    // Debug energy
    static int counter = 0;
    if (counter++ % 60 == 0) {
        float totalEnergy = 0.0f;
        int activeCount = 0;
        
        for (size_t i = 0; i < h_modeStates.size(); i++) {
            float energy = std::abs(h_modeStates[i]);
            if (energy > 0.0001f) {
                totalEnergy += energy;
                activeCount++;
            }
        }
        
        // Also debug which modes are most active
        if (activeCount > 0) {
            float maxEnergy = 0.0f;
            int maxEnergyIdx = -1;
            
            for (size_t i = 0; i < h_modeStates.size(); i++) {
                float energy = std::abs(h_modeStates[i]);
                if (energy > maxEnergy) {
                    maxEnergy = energy;
                    maxEnergyIdx = i;
                }
            }
            
            if (maxEnergyIdx >= 0) {
                int circumferentialModes = kernelParams->circumferentialModes;
                int n = (maxEnergyIdx % circumferentialModes) + 1;
                int m = (maxEnergyIdx / circumferentialModes) % kernelParams->axialModes;
                
                std::cout << "Body energy: " << totalEnergy 
                          << " (active modes: " << activeCount << "/" << h_modeStates.size() 
                          << "), strongest mode: " << maxEnergyIdx
                          << " (n=" << n << ", m=" << m << ")" << std::endl;
            }
        } else {
            std::cout << "Body energy: " << totalEnergy 
                      << " (active modes: " << activeCount << "/" << h_modeStates.size() << ")" << std::endl;
        }
    }
    
    // Get mode parameters
    d_modes->copyToHost(h_modes.data());
    
    if (useMixedOutput) {
        // Mix all microphones
        float mixedSignal = 0.0f;
        int activeMicCount = 0;
        
        for (int i = 0; i < static_cast<int>(microphones.size()); ++i) {
            const auto& mic = microphones[i];
            if (!mic.enabled) continue;
            
            // Sample this microphone position
            float sample = sampleMicrophonePosition(mic);
            
            // Update channel value for visualization/monitoring
            if (i < static_cast<int>(audioChannelIndices.size())) {
                audioManager.setChannelValue(audioChannelIndices[i], sample);
            }
            
            // Add to mix
            mixedSignal += sample;
            activeMicCount++;
        }
        
        // Apply mixing formula and master gain
        if (activeMicCount > 0) {
            mixedSignal /= std::sqrt(static_cast<float>(activeMicCount));
            mixedSignal *= masterGain;
            
            // Process audio step with the mixed signal
            audioManager.processAudioStep(timestep, mixedSignal);
        }
    } else {
        // Use only the first active microphone
        for (int i = 0; i < static_cast<int>(microphones.size()); ++i) {
            const auto& mic = microphones[i];
            if (!mic.enabled) continue;
            
            // Sample this microphone position
            float sample = sampleMicrophonePosition(mic);
            
            // Update channel value for visualization/monitoring
            if (i < static_cast<int>(audioChannelIndices.size())) {
                audioManager.setChannelValue(audioChannelIndices[i], sample);
            }
            
            // Process just this single microphone
            audioManager.processAudioStep(timestep, sample * masterGain);
            break; // Only use the first active microphone
        }
    }
}

float BodyComponent::sampleMicrophonePosition(const BodyVirtualMicrophone& mic) {
    float sample = 0.0f;
    
    // Calculate circumferential and axial modes for indexing
    int circumferentialModes = kernelParams->circumferentialModes;
    int axialModes = kernelParams->axialModes;
    
    // Calculate sample by summing modal contributions
    for (size_t i = 0; i < h_modeStates.size(); i++) {
        // Calculate mode numbers (n,m)
        int n = (i % circumferentialModes) + 1;  // Circumferential (1,2,3,...)
        int m = (i / circumferentialModes) % axialModes;  // Axial (0,1,2,...)
        
        // Get mode state
        float state = h_modeStates[i];
        float amplitude = h_modes[i].amplitude;
        
        // Skip very small states to optimize
        if (std::abs(state) < 0.0001f) continue;
        
        // Calculate spatial weighting based on microphone position
        float theta = mic.position.x * 2.0f * M_PI; // Angular position 
        float height = mic.height;                   // Height position [0-1]
        
        // Compute proper mode shape - different for m=0 vs m>0
        float spatialWeight;
        if (m == 0) {
            // Circumferential mode (no height dependence)
            spatialWeight = std::cos(n * theta);
        } else {
            // Height-dependent mode
            spatialWeight = std::cos(n * theta) * std::sin(m * M_PI * height);
            
            // Boost height-dependent modes to make them more audible
            spatialWeight *= 5.0f;
        }
        
        // Add contribution
        sample += amplitude * state * spatialWeight;
    }
    
    return sample * mic.gain;
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
    auto d_inputExcitation = memoryManager.allocateBuffer<float>(excitation.size());
    d_inputExcitation->copyFromHost(excitation.data());
    
    // Apply excitation to all modes
    ::drumforge::exciteAllModes(d_excitation->get(), d_inputExcitation->get(), kernelParams->numModes);
    
    std::cout << "Applied custom excitation to body modes" << std::endl;
}

void BodyComponent::setupDefaultModes() {
    // Set up parameters
    kernelParams->numModes = 32;  // Total modes
    kernelParams->circumferentialModes = 4;  // n values: 1, 2, 3, 4
    kernelParams->axialModes = 8;  // m values: 0, 1, 2, 3, 4, 5, 6, 7
    
    std::cout << "Set up default mode parameters: " 
              << kernelParams->numModes << " modes total ("
              << kernelParams->circumferentialModes << " circumferential, "
              << kernelParams->axialModes << " axial)" << std::endl;
}

void BodyComponent::setupMaterialPreset(const std::string& material) {
    // Use the helper function to set up the material preset
    ::drumforge::setupMaterialPreset(*kernelParams, material);
    
    std::cout << "Applied material preset for '" << material << "'" << std::endl;
}

void BodyComponent::reportModeFrequencies() {
    // Get mode parameters from device if needed
    if (h_modes.empty() || h_modes.size() != static_cast<size_t>(kernelParams->numModes)) {
        h_modes.resize(kernelParams->numModes);
        if (d_modes) {
            d_modes->copyToHost(h_modes.data());
        }
    }
    
    std::cout << "==== Mode Frequency Report ====" << std::endl;
    std::cout << "Radius: " << radius << ", Height: " << height 
              << ", Thickness: " << thickness << ", Material: " << material << std::endl;
    
    // Calculate mode indices for reporting
    int circumferentialModes = kernelParams->circumferentialModes;
    int axialModes = kernelParams->axialModes;
    
    // Report a selection of modes
    for (size_t i = 0; i < std::min(size_t(16), h_modes.size()); i++) {
        int n = (i % circumferentialModes) + 1;
        int m = (i / circumferentialModes) % axialModes;
        
        std::cout << "Mode " << i << " (n=" << n << ", m=" << m << "): " 
                  << "Freq=" << h_modes[i].frequency << " Hz, "
                  << "Amp=" << h_modes[i].amplitude << ", "
                  << "Decay=" << h_modes[i].decay << " sec" << std::endl;
    }
    std::cout << "============================" << std::endl;
}

//-----------------------------------------------------------------------------
// Parameter Setters
//-----------------------------------------------------------------------------

void BodyComponent::setMaterial(const std::string& newMaterial) {
    if (material == newMaterial) return;  // No change needed
    
    // Update the component's material value
    material = newMaterial;
    
    // Update material parameters
    setupMaterialPreset(material);
    
    // Reinitialize the modes with the new material parameters
    if (d_modes) {
        initializeModes(d_modes->get(), *kernelParams);
        
        // Get updated mode parameters for debugging
        h_modes.resize(kernelParams->numModes);
        d_modes->copyToHost(h_modes.data());
        
        // Report updated frequencies
        reportModeFrequencies();
    }
    
    std::cout << "Body material updated to " << material << std::endl;
}

void BodyComponent::setHeight(float newHeight) {
    if (newHeight <= 0.0f) {
        std::cerr << "Error: Height must be positive" << std::endl;
        return;
    }
    
    if (std::abs(height - newHeight) < 0.001f) return;  // No significant change
    
    // Update the component's height
    height = newHeight;
    kernelParams->height = height;
    
    // Reinitialize the modes with the new height
    if (d_modes) {
        initializeModes(d_modes->get(), *kernelParams);
        
        // Get updated mode parameters for debugging
        h_modes.resize(kernelParams->numModes);
        d_modes->copyToHost(h_modes.data());
        
        // Report updated frequencies
        reportModeFrequencies();
    }
    
    std::cout << "Body height updated to " << height << std::endl;
}

void BodyComponent::setThickness(float newThickness) {
    if (newThickness <= 0.0f) {
        std::cerr << "Error: Thickness must be positive" << std::endl;
        return;
    }
    
    if (std::abs(thickness - newThickness) < 0.0001f) return;  // No significant change
    
    // Update the component's thickness
    thickness = newThickness;
    kernelParams->thickness = thickness;
    
    // Reinitialize the modes with the new thickness
    if (d_modes) {
        initializeModes(d_modes->get(), *kernelParams);
        
        // Get updated mode parameters for debugging
        h_modes.resize(kernelParams->numModes);
        d_modes->copyToHost(h_modes.data());
        
        // Report updated frequencies
        reportModeFrequencies();
    }
    
    std::cout << "Body thickness updated to " << thickness << std::endl;
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
    addMicrophone(0.25f, 0.0f, 0.7f, 0.8f, "Right");  // Right side, higher
    addMicrophone(0.5f, 0.0f, 0.5f, 0.8f, "Back");   // Back center
    addMicrophone(0.75f, 0.0f, 0.3f, 0.8f, "Left");   // Left side, lower
    
    audioChannelsInitialized = false;
}

const std::vector<float>& BodyComponent::getModeStates() const {
    // Copy the current states from device to host for CPU access
    auto& nonConstThis = const_cast<BodyComponent&>(*this);
    auto& nonConstStates = const_cast<std::vector<float>&>(h_modeStates);
    
    if (d_modeStates) {
        d_modeStates->copyToHost(nonConstStates.data());
    }
    
    return h_modeStates;
}

} // namespace drumforge