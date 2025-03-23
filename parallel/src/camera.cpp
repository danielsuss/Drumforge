#include "camera.h"
#include <glm/gtc/matrix_transform.hpp>
#include <iostream>

namespace drumforge {

Camera::Camera(
    glm::vec3 position,
    glm::vec3 target,
    glm::vec3 worldUp,
    float fov,
    float aspectRatio,
    float nearPlane,
    float farPlane)
    : position(position)
    , target(target)
    , worldUp(glm::normalize(worldUp))
    , fieldOfView(fov)
    , aspectRatio(aspectRatio)
    , nearPlane(nearPlane)
    , farPlane(farPlane)
    , moveSpeed(10.0f) {
    
    // Initialize camera vectors
    updateCameraVectors();
    
    std::cout << "Camera initialized at position: (" 
              << position.x << ", " << position.y << ", " << position.z 
              << "), looking at: (" 
              << target.x << ", " << target.y << ", " << target.z << ")" 
              << std::endl;
}

void Camera::updateCameraVectors() {
    // Calculate the new forward vector (normalized direction to target)
    glm::vec3 forward = glm::normalize(target - position);
    
    // Calculate the right vector (cross product of forward and world up)
    right = glm::normalize(glm::cross(forward, worldUp));
    
    // Recalculate the up vector (cross product of right and forward)
    up = glm::normalize(glm::cross(right, forward));
}

glm::mat4 Camera::getViewMatrix() const {
    // Create look-at matrix using the camera position, target, and up vector
    return glm::lookAt(position, target, up);
}

glm::mat4 Camera::getProjectionMatrix() const {
    // Create perspective projection matrix
    return glm::perspective(
        glm::radians(fieldOfView),  // Convert FOV from degrees to radians
        aspectRatio,                // Aspect ratio (width/height)
        nearPlane,                  // Near clipping plane
        farPlane                    // Far clipping plane
    );
}

void Camera::setPosition(const glm::vec3& newPosition) {
    position = newPosition;
    updateCameraVectors();
}

void Camera::setTarget(const glm::vec3& newTarget) {
    target = newTarget;
    updateCameraVectors();
}

void Camera::setFieldOfView(float fov) {
    fieldOfView = fov;
    // Clamp field of view to reasonable range (1 to 120 degrees)
    if (fieldOfView < 1.0f) fieldOfView = 1.0f;
    if (fieldOfView > 120.0f) fieldOfView = 120.0f;
}

void Camera::setAspectRatio(float ratio) {
    aspectRatio = ratio;
}

void Camera::moveForward(float distance) {
    // Calculate normalized direction from position to target
    glm::vec3 direction = glm::normalize(target - position);
    
    // Move both position and target along this direction
    glm::vec3 movement = direction * distance;
    position += movement;
    target += movement;
}

void Camera::moveRight(float distance) {
    // Move both position and target along the right vector
    glm::vec3 movement = right * distance;
    position += movement;
    target += movement;
}

void Camera::moveUp(float distance) {
    // Move both position and target along the up vector
    glm::vec3 movement = up * distance;
    position += movement;
    target += movement;
}

void Camera::reset(const glm::vec3& newPosition, const glm::vec3& newTarget) {
    position = newPosition;
    target = newTarget;
    updateCameraVectors();
    
    std::cout << "Camera reset to position: (" 
              << position.x << ", " << position.y << ", " << position.z 
              << "), looking at: (" 
              << target.x << ", " << target.y << ", " << target.z << ")" 
              << std::endl;
}

} // namespace drumforge