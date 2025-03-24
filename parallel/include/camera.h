#ifndef DRUMFORGE_CAMERA_H
#define DRUMFORGE_CAMERA_H

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

namespace drumforge {

/**
 * @brief Camera class for 3D navigation and view control
 * 
 * This class manages the camera position, orientation, and projection
 * settings for 3D visualization. It provides methods for generating
 * view and projection matrices, and for navigating the 3D scene.
 */
class Camera {
private:
    // Camera position and orientation
    glm::vec3 position;     // Camera position in world space
    glm::vec3 target;       // Look-at target point
    glm::vec3 up;           // Up vector
    glm::vec3 right;        // Right vector
    glm::vec3 worldUp;      // World up vector (typically +Y or +Z)
    
    // Camera parameters
    float fieldOfView;      // Field of view in degrees
    float aspectRatio;      // Aspect ratio (width/height)
    float nearPlane;        // Near clipping plane
    float farPlane;         // Far clipping plane
    
    // Movement parameters
    float moveSpeed;        // Movement speed
    
    // Update camera vectors based on orientation
    void updateCameraVectors();

public:
    /**
     * @brief Constructor
     * 
     * @param position Initial camera position
     * @param target Initial target/look-at point
     * @param worldUp World up vector
     * @param fov Field of view in degrees
     * @param aspectRatio Aspect ratio (width/height)
     * @param nearPlane Near clipping plane
     * @param farPlane Far clipping plane
     */
    Camera(
        glm::vec3 position = glm::vec3(0.0f, 0.0f, 10.0f),
        glm::vec3 target = glm::vec3(0.0f, 0.0f, 0.0f),
        glm::vec3 worldUp = glm::vec3(0.0f, 1.0f, 0.0f),
        float fov = 45.0f,
        float aspectRatio = 16.0f/9.0f,
        float nearPlane = 0.1f,
        float farPlane = 1000.0f
    );
    
    /**
     * @brief Get the view matrix
     * 
     * @return glm::mat4 The view matrix for rendering
     */
    glm::mat4 getViewMatrix() const;
    
    /**
     * @brief Get the projection matrix
     * 
     * @return glm::mat4 The projection matrix for rendering
     */
    glm::mat4 getProjectionMatrix() const;
    
    /**
     * @brief Set new camera position
     * 
     * @param newPosition The new position vector
     */
    void setPosition(const glm::vec3& newPosition);
    
    /**
     * @brief Set new target point
     * 
     * @param newTarget The new target vector
     */
    void setTarget(const glm::vec3& newTarget);
    
    /**
     * @brief Set new field of view
     * 
     * @param fov The new field of view in degrees
     */
    void setFieldOfView(float fov);
    
    /**
     * @brief Set new aspect ratio
     * 
     * @param ratio The new aspect ratio (width/height)
     */
    void setAspectRatio(float ratio);
    
    /**
     * @brief Move the camera forward/backward
     * 
     * @param distance Distance to move (positive = forward, negative = backward)
     */
    void moveForward(float distance);
    
    /**
     * @brief Move the camera right/left
     * 
     * @param distance Distance to move (positive = right, negative = left)
     */
    void moveRight(float distance);
    
    /**
     * @brief Move the camera up/down
     * 
     * @param distance Distance to move (positive = up, negative = down)
     */
    void moveUp(float distance);
    
    /**
     * @brief Pan the camera target left/right
     * 
     * @param distance Distance to pan target (positive = right, negative = left)
     */
    void panTargetRight(float distance);
    
    /**
     * @brief Pan the camera target up/down
     * 
     * @param distance Distance to pan target (positive = up, negative = down)
     */
    void panTargetUp(float distance);
    
    /**
     * @brief Reset camera to default position
     * 
     * @param newPosition New camera position
     * @param newTarget New target position
     */
    void reset(const glm::vec3& newPosition, const glm::vec3& newTarget);
    
    // Getters
    glm::vec3 getPosition() const { return position; }
    glm::vec3 getTarget() const { return target; }
    glm::vec3 getForwardDirection() const { return glm::normalize(target - position); }
    glm::vec3 getRightDirection() const { return right; }
    glm::vec3 getUpDirection() const { return up; }
    float getFieldOfView() const { return fieldOfView; }
    float getAspectRatio() const { return aspectRatio; }
    
    // Setter for movement speed
    void setMoveSpeed(float speed) { moveSpeed = speed; }
};

} // namespace drumforge

#endif // DRUMFORGE_CAMERA_H