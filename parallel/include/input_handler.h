#ifndef DRUMFORGE_INPUT_HANDLER_H
#define DRUMFORGE_INPUT_HANDLER_H

#include <GL/glew.h>  // GLEW must be included before any other OpenGL headers
#include <GLFW/glfw3.h>
#include <memory>
#include <glm/glm.hpp>

namespace drumforge {

// Forward declarations
class Camera;

// Forward declaration for membrane component
class MembraneComponent;

/**
 * @brief Class for handling user input
 *
 * This class centralizes input processing for camera movement and basic
 * application control.
 */
class InputHandler {
private:
    GLFWwindow* window;                  // GLFW window reference
    std::shared_ptr<Camera> camera;      // Connected camera
    bool firstMouse;                     // Flag for mouse input initialization
    glm::vec2 lastMousePos;              // Last mouse position
    float cameraMoveSpeed;               // Camera movement speed multiplier
    int windowWidth;                     // Current window width
    int windowHeight;                    // Current window height
    
    // Stored membrane reference for click interactions
    std::weak_ptr<MembraneComponent> membraneComponent;
    
    // Mouse state
    bool mouseLeftPressed;               // Left mouse button state
    
    // Internal input state
    struct {
        bool forward;    // W key
        bool backward;   // S key
        bool left;       // A key
        bool right;      // D key
        bool up;         // E key
        bool down;       // Q key
    } movementKeys;

public:
    /**
     * @brief Constructor
     * 
     * @param window GLFW window to handle input for
     */
    InputHandler(GLFWwindow* window);
    
    /**
     * @brief Destructor
     */
    ~InputHandler();
    
    /**
     * @brief Process input events for the current frame
     * 
     * @param deltaTime Time elapsed since last frame in seconds
     */
    void processInput(float deltaTime);
    
    /**
     * @brief Connect a camera for movement control
     * 
     * @param camera Pointer to the camera to control
     */
    void connectCamera(std::shared_ptr<Camera> camera);
    
    /**
     * @brief Set the camera movement speed
     * 
     * @param speed Movement speed multiplier
     */
    void setCameraMoveSpeed(float speed) { cameraMoveSpeed = speed; }
    
    /**
     * @brief Get the camera movement speed
     * 
     * @return float The current movement speed multiplier
     */
    float getCameraMoveSpeed() const { return cameraMoveSpeed; }
    
    /**
     * @brief Check if the application should close
     * 
     * @return true if the window should close, false otherwise
     */
    bool shouldClose() const;
    
    /**
     * @brief Key callback function for GLFW
     * 
     * Static callback that will be registered with GLFW.
     * The user pointer of the window should be set to the InputHandler instance.
     */
    static void keyCallback(GLFWwindow* window, int key, int scancode, int action, int mods);
    
    /**
     * @brief Mouse button callback function for GLFW
     */
    static void mouseButtonCallback(GLFWwindow* window, int button, int action, int mods);
    
    /**
     * @brief Window resize callback function for GLFW
     */
    static void windowSizeCallback(GLFWwindow* window, int width, int height);
    
    /**
     * @brief Connect a membrane component for click interactions
     * 
     * @param membrane Shared pointer to the membrane component
     */
    void connectMembrane(std::shared_ptr<MembraneComponent> membrane);
    
    /**
     * @brief Process mouse click on membrane
     * 
     * @param mouseX X position of mouse in screen coordinates
     * @param mouseY Y position of mouse in screen coordinates
     */
    void processMembraneClick(double mouseX, double mouseY);
    
    /**
     * @brief Calculate ray from screen coordinates
     * 
     * @param screenX X position in screen coordinates
     * @param screenY Y position in screen coordinates
     * @return Ray origin and direction
     */
    std::pair<glm::vec3, glm::vec3> screenToRay(double screenX, double screenY) const;
};

} // namespace drumforge

#endif // DRUMFORGE_INPUT_HANDLER_H