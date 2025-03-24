#include "input_handler.h"
#include "camera.h"
#include <iostream>

namespace drumforge {

// Constructor
InputHandler::InputHandler(GLFWwindow* window)
    : window(window)
    , camera(nullptr)
    , firstMouse(true)
    , lastMousePos(0.0f, 0.0f)
    , cameraMoveSpeed(10.0f)
    , movementKeys{false, false, false, false, false, false} {
    
    // Register this object with GLFW via user pointer for callbacks
    glfwSetWindowUserPointer(window, this);
    
    // Register static key callback
    glfwSetKeyCallback(window, keyCallback);
    
    std::cout << "InputHandler initialized" << std::endl;
}

// Destructor
InputHandler::~InputHandler() {
    std::cout << "InputHandler destroyed" << std::endl;
}

// Process input events for the current frame
void InputHandler::processInput(float deltaTime) {
    // Only process movement if camera is connected
    if (!camera) {
        return;
    }
    
    // Calculate actual movement speed based on time
    float actualSpeed = cameraMoveSpeed * deltaTime;
    
    // Process movement keys
    if (movementKeys.forward) {
        camera->moveForward(actualSpeed);
    }
    if (movementKeys.backward) {
        camera->moveForward(-actualSpeed);
    }
    if (movementKeys.left) {
        camera->moveRight(-actualSpeed);
    }
    if (movementKeys.right) {
        camera->moveRight(actualSpeed);
    }
    if (movementKeys.up) {
        camera->moveUp(actualSpeed);
    }
    if (movementKeys.down) {
        camera->moveUp(-actualSpeed);
    }
}

// Connect a camera for movement control
void InputHandler::connectCamera(std::shared_ptr<Camera> cam) {
    camera = cam;
    std::cout << "Camera connected to InputHandler" << std::endl;
}

// Check if the application should close
bool InputHandler::shouldClose() const {
    return glfwWindowShouldClose(window);
}

// Static key callback function for GLFW
void InputHandler::keyCallback(GLFWwindow* window, int key, int scancode, int action, int mods) {
    // Get the InputHandler instance from the window user pointer
    InputHandler* handler = static_cast<InputHandler*>(glfwGetWindowUserPointer(window));
    
    if (!handler) {
        return;
    }
    
    // Handle key press and release events
    const bool pressed = (action == GLFW_PRESS);
    const bool released = (action == GLFW_RELEASE);
    
    if (pressed || released) {
        // Update movement keys state
        switch (key) {
            case GLFW_KEY_W:
                handler->movementKeys.forward = pressed;
                break;
            case GLFW_KEY_S:
                handler->movementKeys.backward = pressed;
                break;
            case GLFW_KEY_A:
                handler->movementKeys.left = pressed;
                break;
            case GLFW_KEY_D:
                handler->movementKeys.right = pressed;
                break;
            case GLFW_KEY_E:
                handler->movementKeys.up = pressed;
                break;
            case GLFW_KEY_Q:
                handler->movementKeys.down = pressed;
                break;
            case GLFW_KEY_ESCAPE:
                if (pressed) {
                    glfwSetWindowShouldClose(window, GLFW_TRUE);
                    std::cout << "ESC pressed, application will close" << std::endl;
                }
                break;
        }
    }
    
    // Handle other key presses (just once when pressed, not continuously)
    if (action == GLFW_PRESS) {
        switch (key) {
            case GLFW_KEY_R:
                // Reset camera to default position (if connected)
                if (handler->camera) {
                    // Default values - could be parameterized in the future
                    handler->camera->reset(
                        glm::vec3(0.0f, 0.0f, 10.0f), 
                        glm::vec3(0.0f, 0.0f, 0.0f)
                    );
                    std::cout << "Camera reset to default position" << std::endl;
                }
                break;
            case GLFW_KEY_F:
                // Toggle full screen (example of additional functionality)
                // Implementation would depend on specific window management needs
                std::cout << "F pressed - fullscreen toggle would go here" << std::endl;
                break;
            case GLFW_KEY_H:
                // Show help
                std::cout << "\n=== Keyboard Controls ===\n";
                std::cout << "W/S - Move forward/backward\n";
                std::cout << "A/D - Move left/right\n";
                std::cout << "Q/E - Move down/up\n";
                std::cout << "R   - Reset camera\n";
                std::cout << "ESC - Exit application\n";
                std::cout << "H   - Show this help message\n";
                std::cout << "========================\n" << std::endl;
                break;
        }
    }
}

} // namespace drumforge