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
    
    // GUI state variables
    struct {
        // SimulationManager params
        float timeScale;
        int gridSize;  // X, Y, Z will be the same
        
        // Membrane params
        float radius;
        float tension;
        float damping;
    } guiState;
    
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
    
    // Render all GUI controls
    void renderGUI(SimulationManager& simManager, std::shared_ptr<MembraneComponent> membrane);
    
    // Apply any parameter changes to the simulation
    void applyChanges(SimulationManager& simManager, std::shared_ptr<MembraneComponent> membrane);
};

} // namespace drumforge

#endif // DRUMFORGE_GUI_MANAGER_H