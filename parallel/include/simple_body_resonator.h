#ifndef DRUMFORGE_SIMPLE_BODY_RESONATOR_H
#define DRUMFORGE_SIMPLE_BODY_RESONATOR_H

#include "component_interface.h"
#include "simulation_manager.h"
#include "audio_manager.h"
#include <memory>
#include <vector>
#include <string>
#include <array>
#include <glm/glm.hpp>

namespace drumforge {

/**
 * A simple resonator model for the drum shell that enhances the sound
 * of the membrane using a bank of resonant filters.
 */
class SimpleBodyResonator : 
    public ComponentInterface,
    public std::enable_shared_from_this<SimpleBodyResonator> {
private:
    // Resonant filter implementation
    struct ResonantFilter {
        float frequency;    // Center frequency in Hz
        float resonance;    // Q factor (resonance sharpness)
        float gain;         // Output gain of this filter
        
        // Filter state variables
        float x1 = 0.0f;    // Input history x[n-1]
        float x2 = 0.0f;    // Input history x[n-2]
        float y1 = 0.0f;    // Output history y[n-1]
        float y2 = 0.0f;    // Output history y[n-2]
    };
    
    // Reference to simulation manager
    SimulationManager& simulationManager;
    
    // Physical parameters
    float radius;          // Body radius (same as membrane radius)
    float height;          // Body height relative to radius
    float thickness;       // Shell thickness relative to radius
    float damping;         // Overall decay time
    float masterGain;      // Overall output level
    std::string material;  // Shell material name
    
    // Resonant filters
    std::vector<ResonantFilter> filters;
    
    // Component name
    std::string name;
    
    // Input from membrane
    float lastImpactStrength = 0.0f;
    bool impactOccurred = false;
    float impactDecay = 0.0f;
    
    // Audio output channel index
    int audioChannelIndex = -1;
    
    // Audio sample rate
    float sampleRate = 44100.0f;
    
    // Material presets - store base frequencies for different materials
    struct MaterialPreset {
        std::string name;
        std::array<float, 5> baseFreqs;  // Base frequencies
        std::array<float, 5> resonances; // Resonance values (Q)
        std::array<float, 5> gains;      // Gain values
        float dampingFactor;             // Material-specific damping
    };
    
    std::vector<MaterialPreset> materialPresets;
    
    // Private methods
    void initializeMaterialPresets();
    void setupFiltersForMaterial(const std::string& material);
    void adjustFiltersForDimensions();
    float processSample(float input);
    
public:
    // Constructor
    SimpleBodyResonator(
        const std::string& name,
        float radius,             // Should match membrane radius
        float height = 0.4f,      // Default height as 40% of radius
        float thickness = 0.01f,  // Default thickness as 1% of radius
        const std::string& material = "Maple"
    );
    
    // Deleted copy constructor and assignment operator
    SimpleBodyResonator(const SimpleBodyResonator&) = delete;
    SimpleBodyResonator& operator=(const SimpleBodyResonator&) = delete;
    
    // Destructor
    ~SimpleBodyResonator();
    
    // ComponentInterface methods
    void initialize() override;
    void update(float timestep) override;
    void prepareForVisualization() override;
    CouplingData getInterfaceData() override;
    void setCouplingData(const CouplingData& data) override;
    std::string getName() const override;
    float calculateStableTimestep() const override;
    DimensionRequirement getDimensionRequirement() const override {
        return DimensionRequirement::DIMENSION_3D;
    }
    
    // Audio-related methods
    void initializeAudioChannels() override;
    void updateAudio(float timestep) override;
    bool hasAudio() const override { return true; }
    
    // Simple resonator-specific methods
    void reset();
    
    // Parameter setters
    void setMaterial(const std::string& newMaterial);
    void setHeight(float newHeight);
    void setThickness(float newThickness);
    void setDamping(float newDamping);
    void setMasterGain(float gain);
    
    // Process a single sample through all filters
    float processInput(float input);
    
    // Accessors
    float getRadius() const { return radius; }
    float getHeight() const { return height; }
    float getThickness() const { return thickness; }
    float getDamping() const { return damping; }
    float getMasterGain() const { return masterGain; }
    const std::string& getMaterial() const { return material; }
    int getFilterCount() const { return filters.size(); }
};

} // namespace drumforge

#endif // DRUMFORGE_SIMPLE_BODY_RESONATOR_H