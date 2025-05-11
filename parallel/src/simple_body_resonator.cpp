#include "simple_body_resonator.h"
#include "visualization_manager.h"
#include <iostream>
#include <cmath>
#include <algorithm>
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace drumforge {

//-----------------------------------------------------------------------------
// Constructor & Destructor
//-----------------------------------------------------------------------------

SimpleBodyResonator::SimpleBodyResonator(
    const std::string& name,
    float radius,
    float height,
    float thickness,
    const std::string& material)
    : simulationManager(SimulationManager::getInstance())
    , radius(radius)
    , height(height)
    , thickness(thickness)
    , damping(0.05f)  // Default damping
    , masterGain(0.6f)  // Default master gain
    , material(material)
    , name(name) {
    
    // Initialize material presets
    initializeMaterialPresets();
    
    // Set up filters for the selected material
    setupFiltersForMaterial(material);
    
    // Adjust filters based on dimensions
    adjustFiltersForDimensions();
    
    std::cout << "SimpleBodyResonator '" << name << "' created" << std::endl;
    std::cout << "  Dimensions: radius=" << radius 
              << ", height=" << height
              << ", thickness=" << thickness << std::endl;
    std::cout << "  Material: " << material << std::endl;
}

SimpleBodyResonator::~SimpleBodyResonator() {
    std::cout << "SimpleBodyResonator '" << name << "' destroyed" << std::endl;
    
    // Clean up OpenGL resources if they were created
    if (vaoId != 0) {
        glDeleteVertexArrays(1, &vaoId);
        vaoId = 0;
    }
    
    if (eboId != 0) {
        glDeleteBuffers(1, &eboId);
        eboId = 0;
    }
}

//-----------------------------------------------------------------------------
// ComponentInterface Implementation
//-----------------------------------------------------------------------------

void SimpleBodyResonator::initialize() {
    std::cout << "Initializing SimpleBodyResonator..." << std::endl;
    
    // Get sample rate from audio manager
    sampleRate = AudioManager::getInstance().getSampleRate();
    
    // Reset all filter states
    reset();
    
    std::cout << "SimpleBodyResonator initialized successfully" << std::endl;
}

void SimpleBodyResonator::update(float timestep) {
    // Update impact decay
    if (impactOccurred) {
        impactDecay -= timestep;
        if (impactDecay <= 0.0f) {
            impactOccurred = false;
            impactDecay = 0.0f;
        }
    }
}

void SimpleBodyResonator::prepareForVisualization() {
    // No dynamic updates needed for the body visualization
    // The body is represented by a static cylinder shape
}

CouplingData SimpleBodyResonator::getInterfaceData() {
    // The resonator doesn't send coupling data back to the membrane
    return CouplingData();
}

void SimpleBodyResonator::setCouplingData(const CouplingData& data) {
    if (data.hasImpact) {
        // Store the impact strength for audio processing
        lastImpactStrength = data.impactStrength;
        impactOccurred = true;
        impactDecay = 0.1f;  // 100ms decay time for impact excitation
        
        std::cout << "Body resonator received impact: strength=" 
                  << data.impactStrength << ", position=("
                  << data.impactPosition.x << ", " 
                  << data.impactPosition.y << ")" << std::endl;
        
        // Temporarily increase resonance of filters based on impact strength
        float resonanceBoost = data.impactStrength * 0.1f;
        for (auto& filter : filters) {
            filter.resonance = std::min(filter.resonance + resonanceBoost, 0.999f);
        }
    }
}

std::string SimpleBodyResonator::getName() const {
    return name;
}

float SimpleBodyResonator::calculateStableTimestep() const {
    // The resonator doesn't affect the simulation timestep
    return 2.0f / 1.0f;  // Default to 60 Hz simulation rate
}

//-----------------------------------------------------------------------------
// Visualization Methods (added from BodyComponent)
//-----------------------------------------------------------------------------

void SimpleBodyResonator::initializeVisualization(VisualizationManager& visManager) {
    // Check if already initialized
    if (vaoId != 0) {
        return;
    }
    
    std::cout << "Initializing visualization for SimpleBodyResonator '" << name << "'..." << std::endl;
    
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
        
        std::cout << "Visualization for SimpleBodyResonator '" << name << "' initialized successfully" << std::endl;
    }
    catch (const std::exception& e) {
        std::cerr << "Error initializing visualization: " << e.what() << std::endl;
    }
}

void SimpleBodyResonator::visualize(VisualizationManager& visManager) {
    // Get material color based on the current material type
    glm::vec3 materialColor;
    
    if (material == "Maple") {
        materialColor = glm::vec3(0.7f, 0.4f, 0.2f); // Light brown for maple
    }
    else if (material == "Birch") {
        materialColor = glm::vec3(0.8f, 0.6f, 0.3f); // Lighter yellowish brown for birch
    }
    else if (material == "Mahogany") {
        materialColor = glm::vec3(0.6f, 0.3f, 0.2f); // Darker reddish brown for mahogany
    }
    else if (material == "Metal") {
        materialColor = glm::vec3(0.8f, 0.8f, 0.85f); // Silvery gray for metal
    }
    else if (material == "Acrylic") {
        materialColor = glm::vec3(0.8f, 0.9f, 1.0f); // Translucent blue-ish for acrylic
    }
    else {
        materialColor = glm::vec3(0.7f, 0.4f, 0.2f); // Default wood color
    }
    
    // Render the body using a wireframe
    visManager.renderWireframe(
        vaoId,                    // VAO containing geometry
        eboId,                    // EBO containing wireframe indices
        getNumWireframeIndices(), // Number of indices to draw
        materialColor             // Color based on material
    );
}

int SimpleBodyResonator::getNumWireframeIndices() const {
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

void SimpleBodyResonator::initializeAudioChannels() {
    // Get reference to AudioManager
    AudioManager& audioManager = AudioManager::getInstance();
    
    // Remove existing channel if any
    if (audioChannelIndex >= 0) {
        audioManager.removeChannel(audioChannelIndex);
    }
    
    // Register a channel for the resonator
    audioChannelIndex = audioManager.addChannel(
        name + " Resonance",  // Channel name
        masterGain            // Initial gain
    );
    
    // Add to component's channel indices list for proper cleanup
    audioChannelIndices.clear();
    audioChannelIndices.push_back(audioChannelIndex);
    
    audioChannelsInitialized = true;
    std::cout << "Body resonator audio channel initialized" << std::endl;
}

void SimpleBodyResonator::updateAudio(float timestep) {
    // This is called by the SimulationManager in each update
    // No action needed here as the resonator processes audio when
    // input samples are provided via processInput()
}

//-----------------------------------------------------------------------------
// Resonator-specific Methods
//-----------------------------------------------------------------------------

void SimpleBodyResonator::reset() {
    // Reset all filter states
    for (auto& filter : filters) {
        filter.x1 = 0.0f;
        filter.x2 = 0.0f;
        filter.y1 = 0.0f;
        filter.y2 = 0.0f;
    }
    
    // Reset impact state
    lastImpactStrength = 0.0f;
    impactOccurred = false;
    impactDecay = 0.0f;
    
    std::cout << "Body resonator reset" << std::endl;
}

float SimpleBodyResonator::processInput(float input) {
    float output = 0.0f;
    
    // Impact excitation can boost the input signal
    float impactBoost = impactOccurred ? (1.0f + lastImpactStrength * impactDecay * 10.0f) : 1.0f;
    input *= impactBoost;
    
    // Process through all resonant filters
    for (auto& filter : filters) {
        // Calculate filter coefficients (biquad bandpass filter)
        float omega = 2.0f * M_PI * filter.frequency / sampleRate;
        float alpha = sin(omega) * 0.5f * (1.0f - filter.resonance);
        
        // Bandpass filter coefficients
        float b0 = alpha;
        float b1 = 0.0f;
        float b2 = -alpha;
        float a0 = 1.0f + alpha;
        float a1 = -2.0f * cos(omega);
        float a2 = 1.0f - alpha;
        
        // Normalize coefficients
        b0 /= a0;
        b1 /= a0;
        b2 /= a0;
        a1 /= a0;
        a2 /= a0;
        
        // Process the sample
        float y = b0 * input + b1 * filter.x1 + b2 * filter.x2
                - a1 * filter.y1 - a2 * filter.y2;
        
        // Update filter state
        filter.x2 = filter.x1;
        filter.x1 = input;
        filter.y2 = filter.y1;
        filter.y1 = y;
        
        // Add to output with gain
        output += y * filter.gain;
    }
    
    // Apply master gain
    output *= masterGain;
    
    // Apply damping (very simple approach)
    float dampingFactor = pow(0.9999f, sampleRate * damping);
    
    // Use resonator output for audio
    AudioManager& audioManager = AudioManager::getInstance();
    if (audioManager.getIsRecording() && audioChannelIndex >= 0) {
        audioManager.setChannelValue(audioChannelIndex, output);
    }
    
    return output;
}

//-----------------------------------------------------------------------------
// Parameter Setters
//-----------------------------------------------------------------------------

void SimpleBodyResonator::setMaterial(const std::string& newMaterial) {
    if (material == newMaterial) return;  // No change needed
    
    // Update the material
    material = newMaterial;
    
    // Set up new filters
    setupFiltersForMaterial(material);
    
    // Adjust for current dimensions
    adjustFiltersForDimensions();
    
    std::cout << "Body material updated to " << material << std::endl;
}

void SimpleBodyResonator::setHeight(float newHeight) {
    if (newHeight <= 0.0f) {
        std::cerr << "Error: Height must be positive" << std::endl;
        return;
    }
    
    if (std::abs(height - newHeight) < 0.001f) return;  // No significant change
    
    // Update the height
    height = newHeight;
    
    // Adjust filters
    adjustFiltersForDimensions();
    
    std::cout << "Body height updated to " << height << std::endl;
}

void SimpleBodyResonator::setThickness(float newThickness) {
    if (newThickness <= 0.0f) {
        std::cerr << "Error: Thickness must be positive" << std::endl;
        return;
    }
    
    if (std::abs(thickness - newThickness) < 0.0001f) return;  // No significant change
    
    // Update the thickness
    thickness = newThickness;
    
    // Adjust filters
    adjustFiltersForDimensions();
    
    std::cout << "Body thickness updated to " << thickness << std::endl;
}

void SimpleBodyResonator::setDamping(float newDamping) {
    if (newDamping < 0.0f) {
        std::cerr << "Warning: Damping must be non-negative, clamping to 0" << std::endl;
        newDamping = 0.0f;
    }
    
    damping = newDamping;
    std::cout << "Body damping updated to " << damping << std::endl;
}

void SimpleBodyResonator::setMasterGain(float gain) {
    masterGain = gain;
    std::cout << "Body master gain updated to " << masterGain << std::endl;
}

//-----------------------------------------------------------------------------
// Private Helper Methods
//-----------------------------------------------------------------------------

void SimpleBodyResonator::initializeMaterialPresets() {
    // Clear existing presets
    materialPresets.clear();
    
    // Define materials with their characteristic frequencies and properties
    
    // Maple - balanced, warm tone
    MaterialPreset maple;
    maple.name = "Maple";
    maple.baseFreqs = { 120.0f, 280.0f, 450.0f, 850.0f, 1200.0f };
    maple.resonances = { 0.95f, 0.93f, 0.90f, 0.87f, 0.84f };
    maple.gains = { 0.7f, 0.5f, 0.4f, 0.3f, 0.2f };
    maple.dampingFactor = 0.05f;
    materialPresets.push_back(maple);
    
    // Birch - brighter sound with more attack
    MaterialPreset birch;
    birch.name = "Birch";
    birch.baseFreqs = { 140.0f, 320.0f, 520.0f, 950.0f, 1400.0f };
    birch.resonances = { 0.94f, 0.92f, 0.91f, 0.88f, 0.86f };
    birch.gains = { 0.6f, 0.55f, 0.5f, 0.4f, 0.3f };
    birch.dampingFactor = 0.04f;
    materialPresets.push_back(birch);
    
    // Mahogany - warm and deep sound
    MaterialPreset mahogany;
    mahogany.name = "Mahogany";
    mahogany.baseFreqs = { 100.0f, 220.0f, 380.0f, 720.0f, 980.0f };
    mahogany.resonances = { 0.96f, 0.94f, 0.91f, 0.88f, 0.85f };
    mahogany.gains = { 0.8f, 0.6f, 0.5f, 0.3f, 0.2f };
    mahogany.dampingFactor = 0.06f;
    materialPresets.push_back(mahogany);
    
    // Metal - bright with long sustain
    MaterialPreset metal;
    metal.name = "Metal";
    metal.baseFreqs = { 180.0f, 450.0f, 920.0f, 1800.0f, 3200.0f };
    metal.resonances = { 0.98f, 0.97f, 0.96f, 0.95f, 0.93f };
    metal.gains = { 0.5f, 0.6f, 0.7f, 0.6f, 0.5f };
    metal.dampingFactor = 0.02f;
    materialPresets.push_back(metal);
    
    // Acrylic - clear sound with less resonance
    MaterialPreset acrylic;
    acrylic.name = "Acrylic";
    acrylic.baseFreqs = { 160.0f, 400.0f, 780.0f, 1500.0f, 2600.0f };
    acrylic.resonances = { 0.91f, 0.89f, 0.87f, 0.85f, 0.82f };
    acrylic.gains = { 0.55f, 0.5f, 0.45f, 0.4f, 0.35f };
    acrylic.dampingFactor = 0.08f;
    materialPresets.push_back(acrylic);
    
    std::cout << "Initialized " << materialPresets.size() << " material presets" << std::endl;
}

void SimpleBodyResonator::setupFiltersForMaterial(const std::string& material) {
    // Find the requested material preset
    const MaterialPreset* preset = nullptr;
    for (const auto& p : materialPresets) {
        if (p.name == material) {
            preset = &p;
            break;
        }
    }
    
    // If material not found, default to Maple
    if (!preset) {
        std::cout << "Material '" << material << "' not found, using Maple as default" << std::endl;
        for (const auto& p : materialPresets) {
            if (p.name == "Maple") {
                preset = &p;
                break;
            }
        }
    }
    
    // Still no preset? Create filters with default values
    if (!preset) {
        std::cerr << "No material presets available, using generic values" << std::endl;
        filters.clear();
        filters.resize(3);
        filters[0] = { 120.0f, 0.95f, 0.7f, 0.0f, 0.0f, 0.0f, 0.0f };
        filters[1] = { 450.0f, 0.9f, 0.4f, 0.0f, 0.0f, 0.0f, 0.0f };
        filters[2] = { 850.0f, 0.85f, 0.3f, 0.0f, 0.0f, 0.0f, 0.0f };
        return;
    }
    
    // Set damping from material
    damping = preset->dampingFactor;
    
    // Create filters from the preset
    filters.clear();
    for (size_t i = 0; i < preset->baseFreqs.size(); i++) {
        ResonantFilter filter;
        filter.frequency = preset->baseFreqs[i];
        filter.resonance = preset->resonances[i];
        filter.gain = preset->gains[i];
        filter.x1 = 0.0f;
        filter.x2 = 0.0f;
        filter.y1 = 0.0f;
        filter.y2 = 0.0f;
        filters.push_back(filter);
    }
    
    std::cout << "Set up " << filters.size() << " resonant filters for material " 
              << material << " (damping=" << damping << ")" << std::endl;
}

void SimpleBodyResonator::adjustFiltersForDimensions() {
    // Adjust filter frequencies based on dimensions
    
    // Size factor: larger radius = lower frequencies
    float sizeFactor = 5.0f / std::max(radius, 0.1f);
    
    // Height factor: taller drum = more low frequencies
    float heightFactor = height / 0.4f;  // Normalize to default height of 0.4
    
    // Thickness factor: thicker shell = higher frequencies and more resonance
    float thicknessFactor = thickness / 0.01f;  // Normalize to default 0.01
    
    for (auto& filter : filters) {
        // Adjust base frequency based on size
        filter.frequency *= sizeFactor;
        
        // Adjust frequency based on height (more effect on lower frequencies)
        if (filter.frequency < 500.0f) {
            // Lower frequencies are more affected by height
            filter.frequency *= 1.0f / (0.7f + 0.3f * heightFactor);
        } else {
            // Higher frequencies are less affected by height
            filter.frequency *= 1.0f / (0.9f + 0.1f * heightFactor);
        }
        
        // Adjust resonance based on thickness
        float resonanceAdjust = 0.02f * (thicknessFactor - 1.0f);
        filter.resonance = std::min(std::max(filter.resonance + resonanceAdjust, 0.5f), 0.99f);
    }
    
    std::cout << "Adjusted filter frequencies for dimensions: radius=" << radius
              << ", height=" << height << ", thickness=" << thickness << std::endl;
    
    // Debug: Print adjusted frequencies
    std::cout << "Resonant frequencies: ";
    for (size_t i = 0; i < std::min(size_t(5), filters.size()); i++) {
        std::cout << filters[i].frequency << "Hz ";
    }
    if (filters.size() > 5) {
        std::cout << "... (" << filters.size() << " total)";
    }
    std::cout << std::endl;
}

} // namespace drumforge