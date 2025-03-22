#ifndef DRUMFORGE_CAMERA_H
#define DRUMFORGE_CAMERA_H

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <GLFW/glfw3.h>

class Camera {
private:
    // Camera attributes
    glm::vec3 position;
    glm::vec3 target;
    glm::vec3 up;
    glm::vec3 right;
    glm::vec3 worldUp;
    
    // Camera movement speeds
    float moveSpeed;
    float panSpeed;
    
    // Recalculate camera vectors (forward, right, up)
    void updateCameraVectors();

public:
    // Constructor with default parameters
    Camera(
        glm::vec3 position = glm::vec3(0.0f, 0.0f, 0.0f),
        glm::vec3 target = glm::vec3(0.0f, 0.0f, 0.0f),
        glm::vec3 worldUp = glm::vec3(0.0f, 0.0f, 1.0f)
    );
    
    // Get the view matrix for rendering
    glm::mat4 getViewMatrix() const;
    
    // Process keyboard input for camera movement (WASD + QE)
    void processMovement(GLFWwindow* window, float deltaTime);
    
    // Process keyboard input for camera panning (arrow keys)
    void processPanning(GLFWwindow* window, float deltaTime);
    
    // Set camera position
    void setPosition(const glm::vec3& newPosition);
    
    // Set camera target
    void setTarget(const glm::vec3& newTarget);
    
    // Look at a specific point
    void lookAt(const glm::vec3& point);
    
    // Reset camera to default position
    void reset(const glm::vec3& newPosition, const glm::vec3& newTarget);
    
    // Getters
    glm::vec3 getPosition() const { return position; }
    glm::vec3 getTarget() const { return target; }
    glm::vec3 getForward() const { return glm::normalize(target - position); }
    glm::vec3 getRight() const { return right; }
    glm::vec3 getUp() const { return up; }
    
    // Set movement speed
    void setMoveSpeed(float speed) { moveSpeed = speed; }
    void setPanSpeed(float speed) { panSpeed = speed; }
};

#endif // DRUMFORGE_CAMERA_Hs