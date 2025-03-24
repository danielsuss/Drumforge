#include <GL/glew.h>  // GLEW must be included before any other OpenGL headers

#include "input_handler.h"
#include "camera.h"
#include "membrane_component.h"
#include <iostream>
#include <glm/gtc/matrix_transform.hpp>

namespace drumforge {

// Constructor
InputHandler::InputHandler(GLFWwindow* window)
    : window(window)
    , camera(nullptr)
    , firstMouse(true)
    , lastMousePos(0.0f, 0.0f)
    , cameraMoveSpeed(1.0f)
    , windowWidth(1280)
    , windowHeight(720)
    , mouseLeftPressed(false)
    , movementKeys{false, false, false, false, false, false} {
    
    // Register this object with GLFW via user pointer for callbacks
    glfwSetWindowUserPointer(window, this);
    
    // Register callbacks
    glfwSetKeyCallback(window, keyCallback);
    glfwSetMouseButtonCallback(window, mouseButtonCallback);
    glfwSetWindowSizeCallback(window, windowSizeCallback);
    
    // Get initial window size
    glfwGetWindowSize(window, &windowWidth, &windowHeight);
    
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
    float panSpeed = cameraMoveSpeed * 1.0f * deltaTime; // Pan speed
    
    // Process movement keys for WASD/QE
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
    
    // Process arrow keys for target panning
    // These direct checks allow arrow keys to work independently of movement keys
    if (glfwGetKey(window, GLFW_KEY_LEFT) == GLFW_PRESS) {
        camera->panTargetRight(-panSpeed);
    }
    if (glfwGetKey(window, GLFW_KEY_RIGHT) == GLFW_PRESS) {
        camera->panTargetRight(panSpeed);
    }
    if (glfwGetKey(window, GLFW_KEY_UP) == GLFW_PRESS) {
        camera->panTargetUp(panSpeed);
    }
    if (glfwGetKey(window, GLFW_KEY_DOWN) == GLFW_PRESS) {
        camera->panTargetUp(-panSpeed);
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
            // Original WASD controls
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
                
            // Arrow keys for panning
            case GLFW_KEY_UP:
                handler->movementKeys.up = pressed;
                break;
            case GLFW_KEY_DOWN:
                handler->movementKeys.down = pressed;
                break;
            case GLFW_KEY_LEFT:
                handler->movementKeys.left = pressed;
                break;
            case GLFW_KEY_RIGHT:
                handler->movementKeys.right = pressed;
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
                std::cout << "\n=== Controls ===\n";
                std::cout << "W/S      - Move camera forward/backward\n";
                std::cout << "A/D      - Move camera left/right\n";
                std::cout << "Q/E      - Move camera down/up\n";
                std::cout << "Arrows   - Pan view direction (change target point)\n";
                std::cout << "R        - Reset camera\n";
                std::cout << "Mouse    - Left-click on membrane to apply impulse\n";
                std::cout << "ESC      - Exit application\n";
                std::cout << "H        - Show this help message\n";
                std::cout << "========================\n" << std::endl;
                break;
        }
    }
}

// Mouse button callback
void InputHandler::mouseButtonCallback(GLFWwindow* window, int button, int action, int mods) {
    // Get the InputHandler instance from the window user pointer
    InputHandler* handler = static_cast<InputHandler*>(glfwGetWindowUserPointer(window));
    
    if (!handler) {
        return;
    }
    
    // Handle mouse button events
    if (button == GLFW_MOUSE_BUTTON_LEFT) {
        if (action == GLFW_PRESS) {
            // Get cursor position
            double mouseX, mouseY;
            glfwGetCursorPos(window, &mouseX, &mouseY);
            
            // Process membrane click
            handler->processMembraneClick(mouseX, mouseY);
        }
    }
}

// Window resize callback
void InputHandler::windowSizeCallback(GLFWwindow* window, int width, int height) {
    // Get the InputHandler instance from the window user pointer
    InputHandler* handler = static_cast<InputHandler*>(glfwGetWindowUserPointer(window));
    
    if (!handler) {
        return;
    }
    
    // Update stored window dimensions
    handler->windowWidth = width;
    handler->windowHeight = height;
    
    // Update camera aspect ratio if connected
    if (handler->camera) {
        handler->camera->setAspectRatio(static_cast<float>(width) / height);
    }
}

// Connect membrane component
void InputHandler::connectMembrane(std::shared_ptr<MembraneComponent> membrane) {
    membraneComponent = membrane;
    std::cout << "MembraneComponent connected to InputHandler" << std::endl;
}

// Process mouse click on membrane
void InputHandler::processMembraneClick(double mouseX, double mouseY) {
    // Check if camera and membrane are connected
    if (!camera) {
        return;
    }
    
    auto membrane = membraneComponent.lock();
    if (!membrane) {
        return;
    }
    
    // Calculate ray from camera through mouse point
    auto [rayOrigin, rayDirection] = screenToRay(mouseX, mouseY);
    
    // Calculate membrane plane (at z = 0)
    glm::vec3 planeNormal(0.0f, 0.0f, 1.0f);
    glm::vec3 planePoint(0.0f, 0.0f, 0.0f); // Center of the membrane
    
    // Check if ray is parallel to the plane
    float denom = glm::dot(rayDirection, planeNormal);
    if (std::abs(denom) < 0.0001f) {
        return; // Ray is parallel to the plane
    }
    
    // Calculate intersection
    float t = glm::dot(planePoint - rayOrigin, planeNormal) / denom;
    if (t <= 0.0f) {
        return; // Intersection is behind the camera
    }
    
    // Calculate intersection point
    glm::vec3 intersectionPoint = rayOrigin + t * rayDirection;
    
    // Convert to normalized membrane coordinates (0 to 1)
    float membraneRadius = membrane->getRadius();
    float clickX = (intersectionPoint.x + membraneRadius) / (2.0f * membraneRadius);
    float clickY = (intersectionPoint.y + membraneRadius) / (2.0f * membraneRadius);
    
    // Check if click is within the membrane radius
    float centerX = 0.5f;
    float centerY = 0.5f;
    float dx = clickX - centerX;
    float dy = clickY - centerY;
    float distSquared = dx * dx + dy * dy;
    
    if (distSquared <= 0.25f) { // 0.5 * 0.5 = 0.25 is the squared radius in normalized coordinates
        // Apply impulse with a fixed strength
        membrane->applyImpulse(clickX, clickY, 1.0f);
        std::cout << "Applied impulse at membrane coordinates (" << clickX << ", " << clickY << ")" << std::endl;
    }
}

// Calculate ray from screen coordinates
std::pair<glm::vec3, glm::vec3> InputHandler::screenToRay(double screenX, double screenY) const {
    if (!camera) {
        return {glm::vec3(0.0f), glm::vec3(0.0f, 0.0f, -1.0f)};
    }
    
    // Convert from screen coordinates to normalized device coordinates
    // (-1 to 1 for both x and y)
    float ndcX = (2.0f * static_cast<float>(screenX) / windowWidth) - 1.0f;
    float ndcY = 1.0f - (2.0f * static_cast<float>(screenY) / windowHeight); // Y is inverted
    
    // Create clip space position
    glm::vec4 clipCoords(ndcX, ndcY, -1.0f, 1.0f);
    
    // Convert to eye space
    glm::mat4 projMatrix = camera->getProjectionMatrix();
    glm::mat4 invProjMatrix = glm::inverse(projMatrix);
    glm::vec4 eyeCoords = invProjMatrix * clipCoords;
    eyeCoords.z = -1.0f;
    eyeCoords.w = 0.0f;
    
    // Convert to world space
    glm::mat4 viewMatrix = camera->getViewMatrix();
    glm::mat4 invViewMatrix = glm::inverse(viewMatrix);
    glm::vec4 worldCoords = invViewMatrix * eyeCoords;
    
    // Extract ray origin and direction
    glm::vec3 rayOrigin = camera->getPosition();
    glm::vec3 rayDirection = glm::normalize(glm::vec3(worldCoords));
    
    return {rayOrigin, rayDirection};
}

} // namespace drumforge