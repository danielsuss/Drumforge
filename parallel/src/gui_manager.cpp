#include "gui_manager.h"
#include "simulation_manager.h"
#include "membrane_component.h"
#include "membrane_kernels.cuh"
#include "audio_manager.h"

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
    configState.gridSizeZ = 128;
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
    ImGui::SliderInt("Grid Size Z", &configState.gridSizeZ, 64, 256);
    
    // Cell Size Control
    ImGui::SliderFloat("Cell Size", &configState.cellSize, 0.05f, 0.2f, "%.2f");
    
    ImGui::Spacing();
    ImGui::Spacing();
    
    // Membrane Configuration Section
    ImGui::TextColored(ImVec4(0.4f, 1.0f, 0.8f, 1.0f), "Membrane Configuration");
    ImGui::Separator();
    
    // Membrane Controls
    ImGui::SliderFloat("Membrane Radius", &configState.radius, 1.0f, 10.0f, "%.1f");
    
    ImGui::SliderFloat("Tension", &configState.tension, 10.0f, 1000.0f, "%.0f");
    
    ImGui::SliderFloat("Damping", &configState.damping, 0.001f, 0.1f, "%.3f", ImGuiSliderFlags_Logarithmic);
    
    ImGui::Spacing();
    ImGui::Spacing();

    // Body Configuration Section
    ImGui::TextColored(ImVec4(0.8f, 0.6f, 0.2f, 1.0f), "Body Configuration");
    ImGui::Separator();

    // Material selection
    const char* materials[] = { "Maple", "Birch", "Mahogany", "Metal", "Acrylic" };
    static int currentMaterial = 0;

    // Find current material in the list
    for (int i = 0; i < IM_ARRAYSIZE(materials); i++) {
        if (configState.bodyMaterial == materials[i]) {
            currentMaterial = i;
            break;
        }
    }

    // Display material combo box
    if (ImGui::Combo("Shell Material", &currentMaterial, materials, IM_ARRAYSIZE(materials))) {
        configState.bodyMaterial = materials[currentMaterial];
    }
    if (ImGui::IsItemHovered()) {
        ImGui::SetTooltip("Different materials produce different tonal qualities");
    }

    // Body height as proportion of radius
    ImGui::SliderFloat("Shell Height", &configState.bodyHeight, 0.2f, 1.0f, "%.2f");
    if (ImGui::IsItemHovered()) {
        ImGui::SetTooltip("Height of the shell as a proportion of the membrane radius");
    }

    // Shell thickness as proportion of radius
    ImGui::SliderFloat("Shell Thickness", &configState.bodyThickness, 0.005f, 0.05f, "%.3f");
    if (ImGui::IsItemHovered()) {
        ImGui::SetTooltip("Thickness of the shell as a proportion of the membrane radius");
    }

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
    ImGui::SliderFloat("Tension", &runtimeState.tension, 10.0f, 1000.0f, "%.0f");
    
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

    // Add this code to the renderSimulationGUI method in gui_manager.cpp
// Place it after the existing membrane controls section but before the audio recording section

    ImGui::Spacing();
    ImGui::Separator();
    ImGui::TextColored(ImVec4(0.0f, 0.8f, 1.0f, 1.0f), "Microphone Configuration");
    
    if (membrane) {
        // Master gain control
        float masterGain = membrane->getMasterGain();
        if (ImGui::SliderFloat("Master Gain", &masterGain, 0.1f, 20.0f, "%.2f")) {
            membrane->setMasterGain(masterGain);
        }
        
        // Mixed output toggle
        bool useMixedOutput = membrane->getUseMixedOutput();
        if (ImGui::Checkbox("Mix All Microphones", &useMixedOutput)) {
            membrane->setUseMixedOutput(useMixedOutput);
        }
        if (ImGui::IsItemHovered()) {
            ImGui::SetTooltip("When enabled, all microphone signals are mixed together.\nWhen disabled, only the first active microphone is used.");
        }
        
        // Microphone presets
        if (ImGui::TreeNode("Microphone Presets")) {
            if (ImGui::Button("Single Center")) {
                membrane->setupSingleCenterMicrophone();
            }
            
            ImGui::SameLine();
            if (ImGui::Button("Stereo")) {
                membrane->setupStereoMicrophones();
            }
            
            ImGui::SameLine();
            if (ImGui::Button("Quad")) {
                membrane->setupQuadMicrophones();
            }
            
            // Circular arrangement with count control
            static int micCount = 8;
            static float circleRadius = 0.4f;
            
            ImGui::PushItemWidth(60);
            ImGui::InputInt("Count", &micCount, 1, 2);
            micCount = std::max(3, std::min(micCount, 16)); // Limit to sensible range
            ImGui::SameLine();
            
            ImGui::PushItemWidth(60);
            ImGui::SliderFloat("Radius", &circleRadius, 0.1f, 0.49f, "%.2f");
            ImGui::SameLine();
            
            if (ImGui::Button("Circular")) {
                membrane->setupCircularMicrophones(micCount, circleRadius);
            }
            
            ImGui::TreePop();
        }
        
        // Individual microphone controls
        if (ImGui::TreeNode("Microphones")) {
            int micCount = membrane->getMicrophoneCount();
            
            // Add new microphone button
            if (ImGui::Button("Add Microphone")) {
                membrane->addMicrophone(0.5f, 0.5f, 1.0f);
            }
            
            ImGui::SameLine();
            if (ImGui::Button("Clear All")) {
                membrane->clearAllMicrophones();
            }
            
            // List all microphones with their properties
            for (int i = 0; i < micCount; i++) {
                const auto& mic = membrane->getMicrophone(i);
                
                // Create a unique ID for each set of widgets
                ImGui::PushID(i);
                
                bool opened = ImGui::TreeNode((void*)(intptr_t)i, "Microphone %d: %s", i + 1, mic.name.c_str());
                
                // Quick controls on the same line as the tree node
                ImGui::SameLine(ImGui::GetWindowWidth() - 120);
                
                // Enable/disable toggle
                bool enabled = mic.enabled;
                if (ImGui::Checkbox("##Enabled", &enabled)) {
                    membrane->enableMicrophone(i, enabled);
                }
                if (ImGui::IsItemHovered()) {
                    ImGui::SetTooltip("Enable/Disable microphone");
                }
                
                ImGui::SameLine();
                
                // Remove button
                if (ImGui::Button("X")) {
                    membrane->removeMicrophone(i);
                    ImGui::PopID();
                    if (opened) ImGui::TreePop();
                    continue; // Skip the rest for this removed microphone
                }
                if (ImGui::IsItemHovered()) {
                    ImGui::SetTooltip("Remove microphone");
                }
                
                if (opened) {
                    // Position control
                    glm::vec2 position = mic.position;
                    float pos[2] = { position.x, position.y };
                    
                    if (ImGui::SliderFloat2("Position", pos, 0.0f, 1.0f, "%.2f")) {
                        membrane->setMicrophonePosition(i, pos[0], pos[1]);
                    }
                    
                    // Gain control
                    float gain = mic.gain;
                    if (ImGui::SliderFloat("Gain", &gain, 0.0f, 2.0f, "%.2f")) {
                        membrane->setMicrophoneGain(i, gain);
                    }
                    
                    // Visualization of position on a grid
                    ImGui::Text("Position Preview:");
                    ImVec2 canvasPos = ImGui::GetCursorScreenPos();
                    ImVec2 canvasSize(100, 100);
                    ImGui::InvisibleButton("canvas", canvasSize);
                    ImDrawList* drawList = ImGui::GetWindowDrawList();
                    
                    // Draw membrane outline
                    ImVec2 center(canvasPos.x + canvasSize.x * 0.5f, canvasPos.y + canvasSize.y * 0.5f);
                    drawList->AddCircle(center, canvasSize.x * 0.48f, ImColor(150, 150, 150, 255), 0, 2.0f);
                    
                    // Draw microphone position
                    ImVec2 micPos(
                        canvasPos.x + canvasSize.x * pos[0],
                        canvasPos.y + canvasSize.y * pos[1]
                    );
                    drawList->AddCircleFilled(micPos, 4.0f, ImColor(255, 50, 50, 255));
                    
                    ImGui::TreePop();
                }
                
                ImGui::PopID();
            }
            
            ImGui::TreePop();
        }
    }
    else {
        ImGui::TextColored(ImVec4(1.0f, 0.3f, 0.3f, 1.0f), "No membrane component available");
    }

    ImGui::Spacing();
    ImGui::Separator();
    ImGui::TextColored(ImVec4(1.0f, 0.0f, 0.0f, 1.0f), "Audio Recording");

    // Get reference to audio manager
    AudioManager& audioManager = AudioManager::getInstance();
    bool isRecording = audioManager.getIsRecording();

    if (ImGui::Button(isRecording ? "Stop Recording" : "Start Recording")) {
        if (isRecording) {
            audioManager.stopRecording();
            
            // Prompt for file name
            ImGui::OpenPopup("Save WAV File");
        } else {
            audioManager.startRecording();
        }
    }

    // Recording indicator
    if (isRecording) {
        ImGui::SameLine();
        ImGui::TextColored(ImVec4(1.0f, 0.0f, 0.0f, 1.0f), "● Recording");
        ImGui::SameLine();
        
        // Show sample count and equivalent time
        float recordedTime = audioManager.getSampleCount() / 
                            static_cast<float>(audioManager.getSampleRate());
        ImGui::Text("(%zu samples, %.2f seconds)", 
                    audioManager.getSampleCount(), recordedTime);
    }

    // Save dialog
    static char filenameBuffer[128] = "drum_recording.wav";
    if (ImGui::BeginPopupModal("Save WAV File", NULL, ImGuiWindowFlags_AlwaysAutoResize)) {
        ImGui::Text("Enter filename to save WAV file:");
        ImGui::InputText("##filename", filenameBuffer, IM_ARRAYSIZE(filenameBuffer));
        
        if (ImGui::Button("Save")) {
            if (audioManager.writeToWavFile(filenameBuffer)) {
                ImGui::CloseCurrentPopup();
                ImGui::OpenPopup("SaveSuccessful");
            } else {
                ImGui::OpenPopup("SaveFailed");
            }
        }
        
        ImGui::SameLine();
        
        if (ImGui::Button("Cancel")) {
            ImGui::CloseCurrentPopup();
        }
        
        // Success/failure dialogs
        if (ImGui::BeginPopupModal("SaveSuccessful", NULL, ImGuiWindowFlags_AlwaysAutoResize)) {
            ImGui::Text("WAV file saved successfully!");
            if (ImGui::Button("OK")) {
                ImGui::CloseCurrentPopup();
            }
            ImGui::EndPopup();
        }
        
        if (ImGui::BeginPopupModal("SaveFailed", NULL, ImGuiWindowFlags_AlwaysAutoResize)) {
            ImGui::Text("Failed to save WAV file!");
            if (ImGui::Button("OK")) {
                ImGui::CloseCurrentPopup();
            }
            ImGui::EndPopup();
        }
        
        ImGui::EndPopup();
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