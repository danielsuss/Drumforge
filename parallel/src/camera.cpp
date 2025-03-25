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
    , moveSpeed(5.0f) {
    
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
    
    // Apply moveSpeed to the distance
    float actualDistance = distance * moveSpeed;
    
    // Move both position and target along this direction
    glm::vec3 movement = direction * actualDistance;
    position += movement;
    target += movement;
}

void Camera::moveRight(float distance) {
    // Apply moveSpeed to the distance
    float actualDistance = distance * moveSpeed;
    
    // Move both position and target along the right vector
    glm::vec3 movement = right * actualDistance;
    position += movement;
    target += movement;
}

void Camera::moveUp(float distance) {
    // Apply moveSpeed to the distance
    float actualDistance = distance * moveSpeed;
    
    // Move both position and target along the world up vector
    // Using worldUp instead of up ensures consistent vertical movement
    glm::vec3 movement = up * actualDistance;
    position += movement;
    target += movement;
}

void Camera::panTargetRight(float distance) {
    // Apply moveSpeed to the distance (with a smaller factor for rotation)
    float actualDistance = distance * (moveSpeed * 0.5f);
    
    // Calculate the distance from camera to target
    float targetDistance = glm::length(target - position);
    
    // Rotate the target around the camera position along the right vector
    // First, move the target relative to the camera
    glm::vec3 relativeTarget = target - position;
    
    // Create a rotation matrix around the up vector
    float angle = actualDistance / targetDistance;  // Small angle approximation
    glm::mat4 rotation = glm::rotate(glm::mat4(1.0f), angle, worldUp);
    
    // Apply rotation to the relative target
    glm::vec4 rotatedTarget = rotation * glm::vec4(relativeTarget, 1.0f);
    
    // Move the target back to world space
    target = position + glm::vec3(rotatedTarget);
    
    // Update camera vectors
    updateCameraVectors();
}

void Camera::panTargetUp(float distance) {
    // Apply moveSpeed to the distance (with a smaller factor for rotation)
    float actualDistance = distance * (moveSpeed * 0.5f);
    
    // Calculate the distance from camera to target
    float targetDistance = glm::length(target - position);
    
    // Rotate the target around the camera position along the up vector
    // First, move the target relative to the camera
    glm::vec3 relativeTarget = target - position;
    
    // Create a rotation matrix around the right vector
    float angle = actualDistance / targetDistance;  // Small angle approximation
    glm::mat4 rotation = glm::rotate(glm::mat4(1.0f), angle, right);
    
    // Apply rotation to the relative target
    glm::vec4 rotatedTarget = rotation * glm::vec4(relativeTarget, 1.0f);
    
    // Move the target back to world space
    target = position + glm::vec3(rotatedTarget);
    
    // Update camera vectors
    updateCameraVectors();
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