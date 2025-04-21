#ifndef DRUMFORGE_GUI_MANAGER_H
#define DRUMFORGE_GUI_MANAGER_H

#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <memory>
#include "body_component.h"  // Include the complete header instead of forward declaration

// Forward declarations
namespace drumforge {
    class SimulationManager;
    class MembraneComponent;
}

namespace drumforge {

/**
 * @brief Represents the application state for GUI rendering
 */
enum class AppState {
    MAIN_MENU,      // Initial configuration menu
    SIMULATION      // Active simulation
};

/**
 * @brief Manages the Dear ImGui-based GUI for the application
 * 
 * This class handles the setup, rendering, and state management for
 * the application's graphical user interface using Dear ImGui.
 */
class GUIManager {
private:
    // Singleton instance
    static std::unique_ptr<GUIManager> instance;
    
    // Reference to the window
    GLFWwindow* window;
    
    // State tracking
    bool initialized;
    AppState currentState;
    bool stateChanged;  // Flag to track state transitions
    
    // GUI state variables for configuration menu
    struct {
        // SimulationManager params
        float timeScale;
        int gridSizeX;  
        int gridSizeY;
        int gridSizeZ;
        float cellSize;
        
        // Membrane params
        float radius;
        float tension;
        float damping;
        
        // Flags
        bool readyToInitialize;  // Set when user confirms settings
        bool simulationInitialized;  // Set when simulation has been initialized
    } configState;
    
    // GUI state variables for simulation controls
    struct {
        float timeScale;
        float tension;
        float damping;
        bool showDebugInfo;
        
        // Body parameters
        std::string bodyMaterial;
        float bodyHeight;
        float bodyThickness;
        float bodyMasterGain;
    } runtimeState;
    
    // Constructor (private for singleton)
    GUIManager();
    
public:
    // Destructor
    ~GUIManager();
    
    // Prevent copying
    GUIManager(const GUIManager&) = delete;
    GUIManager& operator=(const GUIManager&) = delete;
    
    // Get singleton instance
    static GUIManager& getInstance();
    
    // Initialize the GUI
    bool initialize(GLFWwindow* window);
    
    // Clean up resources
    void shutdown();
    
    // Begin a new frame (call before any ImGui commands)
    void beginFrame();
    
    // Render the GUI (call after all ImGui commands)
    void renderFrame();
    
    // Render the appropriate GUI based on application state
    void renderGUI(SimulationManager& simManager, 
                  std::shared_ptr<MembraneComponent> membrane,
                  std::shared_ptr<BodyComponent> body = nullptr);
    
    // Application state methods
    AppState getState() const { return currentState; }
    void setState(AppState newState);
    bool hasStateChanged() const { return stateChanged; }
    void clearStateChanged() { stateChanged = false; }

    // Check if simulation should be initialized (after config)
    bool shouldInitializeSimulation() const { return configState.readyToInitialize && !configState.simulationInitialized; }
    void setSimulationInitialized() { configState.simulationInitialized = true; }
    
    // Get configuration parameters
    int getConfigGridSizeX() const { return configState.gridSizeX; }
    int getConfigGridSizeY() const { return configState.gridSizeY; }
    int getConfigGridSizeZ() const { return configState.gridSizeZ; }
    float getConfigCellSize() const { return configState.cellSize; }
    float getConfigRadius() const { return configState.radius; }
    float getConfigTension() const { return configState.tension; }
    float getConfigDamping() const { return configState.damping; }
    float getConfigTimeScale() const { return configState.timeScale; }

private:
    // Render specific GUI screens
    void renderMainMenu();
    void renderSimulationGUI(SimulationManager& simManager, 
                            std::shared_ptr<MembraneComponent> membrane,
                            std::shared_ptr<BodyComponent> body);
    
    // Apply runtime parameter changes to the simulation
    void applyRuntimeChanges(SimulationManager& simManager, 
                            std::shared_ptr<MembraneComponent> membrane,
                            std::shared_ptr<BodyComponent> body = nullptr);
};

} // namespace drumforge

#endif // DRUMFORGE_GUI_MANAGER_H