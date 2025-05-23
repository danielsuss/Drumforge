#ifndef DRUMFORGE_GUI_MANAGER_H
#define DRUMFORGE_GUI_MANAGER_H

#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <memory>

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

        // Body params
        std::string bodyMaterial = "Maple"; // Default material
        float bodyHeight = 0.4f;            // Default height as proportion of radius
        float bodyThickness = 0.01f;        // Default thickness as proportion of radius
        
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
    void renderGUI(SimulationManager& simManager, std::shared_ptr<MembraneComponent> membrane);
    
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

    const std::string& getConfigBodyMaterial() const { return configState.bodyMaterial; }
    float getConfigBodyHeight() const { return configState.bodyHeight; }
    float getConfigBodyThickness() const { return configState.bodyThickness; }

private:
    // Render specific GUI screens
    void renderMainMenu();
    void renderSimulationGUI(SimulationManager& simManager, std::shared_ptr<MembraneComponent> membrane);
    
    // Apply runtime parameter changes to the simulation
    void applyRuntimeChanges(SimulationManager& simManager, std::shared_ptr<MembraneComponent> membrane);
};

} // namespace drumforge

#endif // DRUMFORGE_GUI_MANAGER_H