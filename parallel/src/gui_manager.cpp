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
    , initialized(false)
    , currentState(AppState::MAIN_MENU)
    , stateChanged(false) {
    
    // Initialize configuration state with defaults
    configState.timeScale = 1.0f;
    configState.gridSizeX = 128;
    configState.gridSizeY = 128;
    configState.gridSizeZ = 16;
    configState.cellSize = 0.1f;
    configState.radius = 5.0f;
    configState.tension = 100.0f;
    configState.damping = 0.01f;
    configState.readyToInitialize = false;
    configState.simulationInitialized = false;
    
    // Initialize runtime state with defaults
    runtimeState.timeScale = 1.0f;
    runtimeState.tension = 100.0f;
    runtimeState.damping = 0.01f;
    runtimeState.showDebugInfo = false;
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
    
    // Render the appropriate GUI based on current state
    switch (currentState) {
        case AppState::MAIN_MENU:
            renderMainMenu();
            break;
            
        case AppState::SIMULATION:
            renderSimulationGUI(simManager, membrane);
            break;
    }
}

void GUIManager::setState(AppState newState) {
    if (newState != currentState) {
        currentState = newState;
        stateChanged = true;
        std::cout << "Application state changed to: " 
                  << (currentState == AppState::MAIN_MENU ? "MAIN_MENU" : "SIMULATION") 
                  << std::endl;
    }
}

void GUIManager::renderMainMenu() {
    // Set up centered window
    ImGuiIO& io = ImGui::GetIO();
    ImVec2 windowSize = ImVec2(450, 500);
    ImVec2 windowPos = ImVec2((io.DisplaySize.x - windowSize.x) * 0.5f, 
                             (io.DisplaySize.y - windowSize.y) * 0.5f);
    
    ImGui::SetNextWindowPos(windowPos, ImGuiCond_Always);
    ImGui::SetNextWindowSize(windowSize, ImGuiCond_Always);
    
    ImGui::Begin("DrumForge - Setup", nullptr, 
                ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove | 
                ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoBringToFrontOnFocus);
    
    // Title and description
    ImGui::TextColored(ImVec4(1.0f, 0.8f, 0.0f, 1.0f), "DrumForge Initial Configuration");
    ImGui::Spacing();
    ImGui::Spacing();
    
    // Grid Configuration Section
    ImGui::TextColored(ImVec4(0.4f, 0.8f, 1.0f, 1.0f), "Grid Configuration");
    ImGui::Separator();
    
    // Grid Size Controls
    ImGui::SliderInt("Grid Size X", &configState.gridSizeX, 64, 256);
    ImGui::SliderInt("Grid Size Y", &configState.gridSizeY, 64, 256);
    ImGui::SliderInt("Grid Size Z", &configState.gridSizeZ, 16, 64);
    
    // Cell Size Control
    ImGui::SliderFloat("Cell Size", &configState.cellSize, 0.05f, 0.2f, "%.2f");
    
    ImGui::Spacing();
    ImGui::Spacing();
    
    // Membrane Configuration Section
    ImGui::TextColored(ImVec4(0.4f, 1.0f, 0.8f, 1.0f), "Membrane Configuration");
    ImGui::Separator();
    
    // Membrane Controls
    ImGui::SliderFloat("Membrane Radius", &configState.radius, 1.0f, 10.0f, "%.1f");
    
    ImGui::SliderFloat("Tension", &configState.tension, 10.0f, 250.0f, "%.0f");
    
    ImGui::SliderFloat("Damping", &configState.damping, 0.001f, 0.1f, "%.3f", ImGuiSliderFlags_Logarithmic);
    
    ImGui::Spacing();
    ImGui::Spacing();
    
    // Simulation Controls Section
    ImGui::TextColored(ImVec4(1.0f, 0.4f, 0.8f, 1.0f), "Simulation Controls");
    ImGui::Separator();
    
    // Time Scale Control
    ImGui::SliderFloat("Time Scale", &configState.timeScale, 0.1f, 1.0f, "%.1f");
    
    ImGui::Spacing();
    ImGui::Spacing();
    
    // Start Simulation Button
    ImGui::SetCursorPosX((ImGui::GetWindowWidth() - 180) * 0.5f);
    if (ImGui::Button("Start Simulation", ImVec2(180, 40))) {
        configState.readyToInitialize = true;
        setState(AppState::SIMULATION);
    }
    
    // Footer text
    ImGui::Spacing();
    ImGui::Separator();
    ImGui::TextColored(ImVec4(0.7f, 0.7f, 0.7f, 1.0f), 
                       "Press ESC at any time to exit the application");
    
    ImGui::End();
}

void GUIManager::renderSimulationGUI(SimulationManager& simManager, std::shared_ptr<MembraneComponent> membrane) {
    // Sync runtime state with current simulation parameters on first run
    static bool firstRunInSimulation = true;
    if (firstRunInSimulation) {
        const auto& params = simManager.getParameters();
        runtimeState.timeScale = params.timeScale;
        
        if (membrane) {
            runtimeState.tension = membrane->getTension();
            runtimeState.damping = membrane->getDamping();
        }
        
        firstRunInSimulation = false;
    }
    
    // Create main control window
    ImGui::SetNextWindowPos(ImVec2(10, 10), ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowSize(ImVec2(300, 380), ImGuiCond_FirstUseEver);
    ImGui::Begin("DrumForge Controls");
    
    // Title and brief state info
    ImGui::TextColored(ImVec4(1.0f, 0.8f, 0.0f, 1.0f), "DrumForge Physics Controls");
    ImGui::Separator();
    
    // Membrane controls
    ImGui::TextColored(ImVec4(0.0f, 1.0f, 1.0f, 1.0f), "Membrane Properties");
    ImGui::Separator();
    
    // Tension control
    ImGui::SliderFloat("Tension", &runtimeState.tension, 10.0f, 250.0f, "%.0f");
    
    // Damping control
    ImGui::SliderFloat("Damping", &runtimeState.damping, 0.001f, 0.1f, "%.3f", ImGuiSliderFlags_Logarithmic);
    
    ImGui::Spacing();
    
    // Simulation controls
    ImGui::TextColored(ImVec4(1.0f, 0.4f, 0.8f, 1.0f), "Simulation Controls");
    ImGui::Separator();
    
    // Time scale control
    ImGui::SliderFloat("Time Scale", &runtimeState.timeScale, 0.1f, 1.0f, "%.1f");
    
    // Debug info toggle
    ImGui::Checkbox("Show Debug Info", &runtimeState.showDebugInfo);
    
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
        applyRuntimeChanges(simManager, membrane);
    }
    
    ImGui::Spacing();
    
    // Display membrane grid info
    if (runtimeState.showDebugInfo && membrane) {
        ImGui::Separator();
        ImGui::TextColored(ImVec4(0.7f, 0.7f, 0.7f, 1.0f), "Debug Information");
        
        ImGui::Text("Grid Size: %dx%dx%d", 
                   configState.gridSizeX, 
                   configState.gridSizeY, 
                   configState.gridSizeZ);
        ImGui::Text("Cell Size: %.3f", configState.cellSize);
        ImGui::Text("Membrane Size: %dx%d", 
                   membrane->getMembraneWidth(),
                   membrane->getMembraneHeight());
        ImGui::Text("Stable Timestep: %.6f", membrane->calculateStableTimestep());
    }
    
    ImGui::End();
}

void GUIManager::applyRuntimeChanges(SimulationManager& simManager, std::shared_ptr<MembraneComponent> membrane) {
    // Update simulation time scale
    SimulationParameters params = simManager.getParameters();
    params.timeScale = runtimeState.timeScale;
    simManager.updateParameters(params);
    
    // Update membrane parameters
    if (membrane) {
        membrane->setTension(runtimeState.tension);
        membrane->setDamping(runtimeState.damping);
        
        std::cout << "Updated membrane tension to " << runtimeState.tension << std::endl;
        std::cout << "Updated membrane damping to " << runtimeState.damping << std::endl;
    }
    
    std::cout << "Applied runtime parameter changes" << std::endl;
}

} // namespace drumforge