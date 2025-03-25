#include "gui_manager.h"
#include "simulation_manager.h"
#include "membrane_component.h"
#include "membrane_kernels.cuh"

// Include Dear ImGui
#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"

#include <iostream>

namespace drumforge {

// Initialize the singleton instance pointer
std::unique_ptr<GUIManager> GUIManager::instance = nullptr;

GUIManager::GUIManager()
    : window(nullptr)
    , initialized(false) {
    
    // Initialize GUI state with defaults
    guiState.timeScale = 1.0f;
    guiState.gridSize = 64;
    guiState.radius = 5.0f;
    guiState.tension = 100.0f;
    guiState.damping = 0.01f;
}

GUIManager::~GUIManager() {
    shutdown();
}

GUIManager& GUIManager::getInstance() {
    if (!instance) {
        instance = std::unique_ptr<GUIManager>(new GUIManager());
    }
    return *instance;
}

bool GUIManager::initialize(GLFWwindow* window) {
    if (initialized) {
        return true;
    }
    
    this->window = window;
    
    std::cout << "Initializing GUIManager using Dear ImGui..." << std::endl;
    
    // Setup Dear ImGui context
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO();
    
    // Enable keyboard navigation
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;
    
    // Setup platform/renderer bindings
    ImGui_ImplGlfw_InitForOpenGL(window, true);
    ImGui_ImplOpenGL3_Init("#version 330");
    
    // Setup Dear ImGui style
    ImGui::StyleColorsDark();
    
    // Make UI more readable
    ImGuiStyle& style = ImGui::GetStyle();
    style.WindowRounding = 5.0f;
    style.FrameRounding = 3.0f;
    style.FramePadding = ImVec2(8, 4);
    style.ItemSpacing = ImVec2(10, 8);
    
    initialized = true;
    std::cout << "GUIManager initialized successfully" << std::endl;
    return true;
}

void GUIManager::shutdown() {
    if (!initialized) {
        return;
    }
    
    std::cout << "Shutting down GUIManager..." << std::endl;
    
    // Shutdown ImGui
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();
    
    initialized = false;
    window = nullptr;
    
    std::cout << "GUIManager shutdown completed" << std::endl;
}

void GUIManager::beginFrame() {
    if (!initialized) {
        return;
    }
    
    // Start a new ImGui frame
    ImGui_ImplOpenGL3_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();
}

void GUIManager::renderFrame() {
    if (!initialized) {
        return;
    }
    
    // Render ImGui
    ImGui::Render();
    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
}

void GUIManager::renderGUI(SimulationManager& simManager, std::shared_ptr<MembraneComponent> membrane) {
    if (!initialized) {
        return;
    }
    
    // Sync GUI state with current simulation parameters on first run
    static bool firstRun = true;
    if (firstRun) {
        const auto& params = simManager.getParameters();
        guiState.timeScale = params.timeScale;
        guiState.gridSize = params.gridSizeX; // Assuming X, Y, Z are the same
        
        if (membrane) {
            guiState.radius = membrane->getRadius();
            
            // For tension and damping, we'll need to access them differently
            // We'll need to modify the MembraneComponent class to expose these properties
            
            // For now, just use default values from the constructor
            guiState.tension = 100.0f;
            guiState.damping = 0.01f;
        }
        
        firstRun = false;
    }
    
    // Create parameter window
    ImGui::Begin("DrumForge Parameters");
    
    // Simulation settings section
    ImGui::TextColored(ImVec4(1.0f, 1.0f, 0.0f, 1.0f), "Simulation Settings");
    ImGui::Separator();
    
    // Time scale control
    ImGui::SliderFloat("Time Scale", &guiState.timeScale, 0.1f, 10.0f, "%.1f");
    ImGui::TextDisabled("Controls the speed of the simulation");
    
    // Grid size control (just one slider for X, Y, Z)
    int gridSize = guiState.gridSize;
    if (ImGui::SliderInt("Grid Size", &gridSize, 32, 256)) {
        guiState.gridSize = gridSize;
    }
    ImGui::TextDisabled("Note: Grid size changes require a restart");
    
    ImGui::Spacing();
    ImGui::Spacing();
    
    // Membrane settings section
    ImGui::TextColored(ImVec4(0.0f, 1.0f, 1.0f, 1.0f), "Membrane Properties");
    ImGui::Separator();
    
    // Radius control
    ImGui::SliderFloat("Radius", &guiState.radius, 1.0f, 10.0f, "%.1f");
    ImGui::TextDisabled("Physical radius of the drum membrane");
    
    // Tension control
    ImGui::SliderFloat("Tension", &guiState.tension, 10.0f, 1000.0f, "%.0f");
    ImGui::TextDisabled("Higher values create tighter, higher-pitched membranes");
    
    // Damping control
    ImGui::SliderFloat("Damping", &guiState.damping, 0.001f, 0.1f, "%.3f", ImGuiSliderFlags_Logarithmic);
    ImGui::TextDisabled("Controls how quickly vibrations decay");
    
    // Reset button
    ImGui::Spacing();
    if (ImGui::Button("Reset Membrane")) {
        if (membrane) {
            membrane->reset();
        }
    }
    
    // Apply button
    ImGui::SameLine();
    if (ImGui::Button("Apply Changes")) {
        applyChanges(simManager, membrane);
    }
    
    ImGui::End();
}

void GUIManager::applyChanges(SimulationManager& simManager, std::shared_ptr<MembraneComponent> membrane) {
    // Apply simulation manager parameters
    SimulationParameters params = simManager.getParameters();
    params.timeScale = guiState.timeScale;
    
    // Grid size changes would ideally require reinitialization,
    // so we'll just note that in the UI. We still set it in case
    // the simulation manager can handle it.
    params.gridSizeX = guiState.gridSize;
    params.gridSizeY = guiState.gridSize;
    params.gridSizeZ = guiState.gridSize;
    
    simManager.updateParameters(params);
    
    // Apply membrane parameters
    if (membrane) {
        // Apply the new membrane parameters
        membrane->setRadius(guiState.radius);
        membrane->setTension(guiState.tension);
        membrane->setDamping(guiState.damping);
        
        std::cout << "Updated membrane radius to " << guiState.radius << std::endl;
        std::cout << "Updated membrane tension to " << guiState.tension << std::endl;
        std::cout << "Updated membrane damping to " << guiState.damping << std::endl;
    }
    
    std::cout << "Applied parameter changes" << std::endl;
}

} // namespace drumforge