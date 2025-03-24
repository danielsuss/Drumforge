#ifndef DRUMFORGE_VISUALIZATION_MANAGER_H
#define DRUMFORGE_VISUALIZATION_MANAGER_H

#include <GL/glew.h>  // GLEW must be included before any OpenGL headers
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <memory>
#include <string>

namespace drumforge {

// Forward declarations
class Camera;
class InputHandler;
class SimulationManager;
class ComponentInterface;

/**
 * @brief Manages the visualization of the simulation using OpenGL.
 * 
 * This class handles the OpenGL context, window creation, shader management,
 * and basic wireframe rendering for simulation components.
 */
class VisualizationManager {
private:
    // Singleton instance
    static std::unique_ptr<VisualizationManager> instance;
    
    // GLFW window
    GLFWwindow* window;
    
    // Window dimensions
    int windowWidth;
    int windowHeight;
    
    // Wireframe shader program ID
    unsigned int wireframeShaderProgram;
    
    // Camera for navigation
    std::shared_ptr<Camera> camera;
    
    // Input handler for user interaction
    std::shared_ptr<InputHandler> inputHandler;
    
    // GLFW callbacks
    static void framebufferSizeCallback(GLFWwindow* window, int width, int height);
    static void errorCallback(int error, const char* description);
    
    // Shader utilities
    unsigned int compileShader(GLenum type, const char* source);
    unsigned int createShaderProgram(const char* vertexSource, const char* fragmentSource);
    
    // Initialization helpers
    bool initGLFW();
    bool initGLEW();
    bool createShaders();
    
    // Private constructor for singleton
    VisualizationManager();

public:
    // Destructor
    ~VisualizationManager();
    
    // Deleted copy constructor and assignment operator
    VisualizationManager(const VisualizationManager&) = delete;
    VisualizationManager& operator=(const VisualizationManager&) = delete;
    
    // Singleton access
    static VisualizationManager& getInstance();
    
    // Initialization
    bool initialize(int width = 1280, int height = 720, const char* title = "DrumForge");
    
    // Frame management
    void beginFrame();
    void endFrame();
    
    // Render all visualizable components from SimulationManager
    void renderComponents(SimulationManager& simManager);
    
    // Initialize visualization for a specific component
    void initializeComponentVisualization(std::shared_ptr<ComponentInterface> component);
    
    // Resource creation methods
    unsigned int createVertexBuffer(size_t size, const void* data = nullptr);
    unsigned int createVertexArray();
    unsigned int createIndexBuffer(const void* data, size_t size);
    void configureVertexAttributes(unsigned int vao, unsigned int vbo, unsigned int location, 
                                   int size, size_t stride, size_t offset);
    
    // Wireframe rendering method
    void renderWireframe(unsigned int vao, unsigned int ebo, int indexCount, const glm::vec3& color);
    
    // Getters
    GLFWwindow* getWindow() const { return window; }
    int getWidth() const { return windowWidth; }
    int getHeight() const { return windowHeight; }
    std::shared_ptr<Camera> getCamera() const { return camera; }
    std::shared_ptr<InputHandler> getInputHandler() const { return inputHandler; }
    
    // Window status check
    bool shouldClose() const;
    
    // Clean up resources
    void shutdown();
};

} // namespace drumforge

#endif // DRUMFORGE_VISUALIZATION_MANAGER_H