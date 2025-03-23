#ifndef DRUMFORGE_INPUT_HANDLER_H
#define DRUMFORGE_INPUT_HANDLER_H

#include <GLFW/glfw3.h>
#include <memory>
#include <glm/glm.hpp>

namespace drumforge {

// Forward declarations
class Camera;

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
};

} // namespace drumforge

#endif // DRUMFORGE_INPUT_HANDLER_H