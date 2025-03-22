#include "camera.h"

Camera::Camera(glm::vec3 position, glm::vec3 target, glm::vec3 worldUp)
    : position(position)
    , target(target)
    , worldUp(worldUp)
    , moveSpeed(20.0f)
    , panSpeed(30.0f)
{
    updateCameraVectors();
}

void Camera::updateCameraVectors()
{
    // Calculate the new forward vector (normalized)
    glm::vec3 forward = glm::normalize(target - position);
    
    // Calculate the new right vector
    right = glm::normalize(glm::cross(forward, worldUp));
    
    // Recalculate the up vector based on right and forward
    up = glm::normalize(glm::cross(right, forward));
}

glm::mat4 Camera::getViewMatrix() const
{
    return glm::lookAt(position, target, up);
}

void Camera::processMovement(GLFWwindow* window, float deltaTime)
{
    float actualSpeed = moveSpeed * deltaTime;
    
    // Calculate forward vector (normalized direction from position to target)
    glm::vec3 forward = glm::normalize(target - position);
    
    // Calculate displacement vectors based on key presses
    glm::vec3 displacement(0.0f);
    
    // Forward/Backward (W/S)
    if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS) {
        displacement += forward * actualSpeed;
    }
    if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS) {
        displacement -= forward * actualSpeed;
    }
    
    // Left/Right (A/D)
    if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS) {
        displacement -= right * actualSpeed;
    }
    if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS) {
        displacement += right * actualSpeed;
    }
    
    // Up/Down (Q/E)
    if (glfwGetKey(window, GLFW_KEY_Q) == GLFW_PRESS) {
        displacement += worldUp * actualSpeed;
    }
    if (glfwGetKey(window, GLFW_KEY_E) == GLFW_PRESS) {
        displacement -= worldUp * actualSpeed;
    }
    
    // Apply displacement to both position and target to maintain relative orientation
    position += displacement;
    target += displacement;
}

void Camera::processPanning(GLFWwindow* window, float deltaTime)
{
    float actualPanSpeed = panSpeed * deltaTime;
    
    // Calculate the distance from camera to target
    float distance = glm::length(target - position);
    
    // Calculate displacement vectors for panning
    glm::vec3 targetDisplacement(0.0f);
    
    // Pan left/right (Left/Right arrow keys)
    if (glfwGetKey(window, GLFW_KEY_LEFT) == GLFW_PRESS) {
        targetDisplacement -= right * actualPanSpeed;
    }
    if (glfwGetKey(window, GLFW_KEY_RIGHT) == GLFW_PRESS) {
        targetDisplacement += right * actualPanSpeed;
    }
    
    // Pan up/down (Up/Down arrow keys)
    if (glfwGetKey(window, GLFW_KEY_UP) == GLFW_PRESS) {
        targetDisplacement += up * actualPanSpeed;
    }
    if (glfwGetKey(window, GLFW_KEY_DOWN) == GLFW_PRESS) {
        targetDisplacement -= up * actualPanSpeed;
    }
    
    // Only move the target point for panning
    target += targetDisplacement;
    
    // Recalculate camera vectors with new target
    updateCameraVectors();
    
    // Ensure the camera maintains the same distance from the target
    position = target - getForward() * distance;
}

void Camera::setPosition(const glm::vec3& newPosition)
{
    position = newPosition;
    updateCameraVectors();
}

void Camera::setTarget(const glm::vec3& newTarget)
{
    target = newTarget;
    updateCameraVectors();
}

void Camera::lookAt(const glm::vec3& point)
{
    target = point;
    updateCameraVectors();
}

void Camera::reset(const glm::vec3& newPosition, const glm::vec3& newTarget)
{
    position = newPosition;
    target = newTarget;
    updateCameraVectors();
}