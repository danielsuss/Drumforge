#include <GL/glew.h>  // GLEW must be included before any OpenGL headers

#include "visualization_manager.h"
#include "camera.h"
#include "input_handler.h"
#include "simulation_manager.h"
#include "component_interface.h"

#include <glm/gtc/type_ptr.hpp>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace drumforge {

// Singleton instance initialization
std::unique_ptr<VisualizationManager> VisualizationManager::instance = nullptr;

// Shader source code as string literals
const char* vertexShaderSource = R"(
    #version 330 core
    layout (location = 0) in vec3 aPos;
    
    uniform mat4 model;
    uniform mat4 view;
    uniform mat4 projection;
    uniform vec3 color;
    
    out vec3 fragColor;
    
    void main() {
        gl_Position = projection * view * model * vec4(aPos, 1.0);
        
        // Color based on displacement (height)
        float displacement = aPos.z;
        if (displacement > 0.0) {
            // Red for positive displacement
            fragColor = mix(color, vec3(1.0, 0.0, 0.0), displacement * 5.0);
        } else {
            // Blue for negative displacement
            fragColor = mix(color, vec3(0.0, 0.0, 1.0), -displacement * 5.0);
        }
    }
)";

const char* fragmentShaderSource = R"(
    #version 330 core
    in vec3 fragColor;
    out vec4 FragColor;
    
    void main() {
        FragColor = vec4(fragColor, 1.0);
    }
)";

//-----------------------------------------------------------------------------
// GLFW Callbacks
//-----------------------------------------------------------------------------

void VisualizationManager::framebufferSizeCallback(GLFWwindow* window, int width, int height) {
    glViewport(0, 0, width, height);
    
    // Update the viewport dimensions in the visualization manager
    VisualizationManager& visManager = VisualizationManager::getInstance();
    visManager.windowWidth = width;
    visManager.windowHeight = height;
    
    // Update the camera aspect ratio if it exists
    if (visManager.camera) {
        visManager.camera->setAspectRatio(static_cast<float>(width) / height);
    }
}

void VisualizationManager::errorCallback(int error, const char* description) {
    std::cerr << "GLFW Error " << error << ": " << description << std::endl;
}

//-----------------------------------------------------------------------------
// Constructor & Destructor
//-----------------------------------------------------------------------------

VisualizationManager::VisualizationManager()
    : window(nullptr)
    , windowWidth(1280)
    , windowHeight(720)
    , wireframeShaderProgram(0)
    , camera(nullptr)
    , inputHandler(nullptr) {
}

VisualizationManager::~VisualizationManager() {
    shutdown();
}

//-----------------------------------------------------------------------------
// Singleton Access
//-----------------------------------------------------------------------------

VisualizationManager& VisualizationManager::getInstance() {
    if (!instance) {
        instance = std::unique_ptr<VisualizationManager>(new VisualizationManager());
    }
    return *instance;
}

//-----------------------------------------------------------------------------
// Initialization
//-----------------------------------------------------------------------------

bool VisualizationManager::initialize(int width, int height, const char* title) {
    std::cout << "Initializing VisualizationManager..." << std::endl;
    
    // Store window dimensions
    windowWidth = width;
    windowHeight = height;
    
    // Initialize GLFW
    if (!initGLFW()) {
        std::cerr << "Failed to initialize GLFW" << std::endl;
        return false;
    }
    
    // Create a window
    window = glfwCreateWindow(width, height, title, nullptr, nullptr);
    if (!window) {
        std::cerr << "Failed to create GLFW window" << std::endl;
        glfwTerminate();
        return false;
    }
    
    // Make the OpenGL context current
    glfwMakeContextCurrent(window);
    
    // Set up callbacks
    glfwSetFramebufferSizeCallback(window, framebufferSizeCallback);
    
    // Initialize GLEW
    if (!initGLEW()) {
        std::cerr << "Failed to initialize GLEW" << std::endl;
        return false;
    }
    
    // Set up viewport
    glViewport(0, 0, width, height);
    
    // Enable depth testing
    glEnable(GL_DEPTH_TEST);
    
    // Create shaders
    if (!createShaders()) {
        std::cerr << "Failed to create shaders" << std::endl;
        return false;
    }
    
    // Create camera
    camera = std::make_shared<Camera>(
        glm::vec3(0.0f, 0.0f, 5.0f),  // Position
        glm::vec3(0.0f, 0.0f, 0.0f),  // Target
        glm::vec3(0.0f, 1.0f, 0.0f),  // Up vector
        45.0f,                        // FOV
        static_cast<float>(width) / height  // Aspect ratio
    );
    
    // Create input handler and connect it to the camera
    inputHandler = std::make_shared<InputHandler>(window);
    inputHandler->connectCamera(camera);
    
    std::cout << "VisualizationManager initialized successfully" << std::endl;
    return true;
}

bool VisualizationManager::initGLFW() {
    // Set error callback
    glfwSetErrorCallback(errorCallback);
    
    // Initialize GLFW
    if (!glfwInit()) {
        std::cerr << "Failed to initialize GLFW" << std::endl;
        return false;
    }
    
    // Set window hints for OpenGL version
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    
    // Important for CUDA interop: make sure we create a window with a shared context
    // This ensures the OpenGL context can be used with CUDA
    glfwWindowHint(GLFW_OPENGL_DEBUG_CONTEXT, GLFW_TRUE);
    glfwWindowHint(GLFW_VISIBLE, GLFW_TRUE);
    
    std::cout << "GLFW initialized with OpenGL 3.3 core profile" << std::endl;
    return true;
}

bool VisualizationManager::initGLEW() {
    // Initialize GLEW
    glewExperimental = GL_TRUE;
    GLenum err = glewInit();
    if (err != GLEW_OK) {
        std::cerr << "GLEW Error: " << glewGetErrorString(err) << std::endl;
        return false;
    }
    
    // Clear any error that might have occurred during GLEW initialization
    // (this is normal, as GLEW sometimes generates a GL_INVALID_ENUM error)
    glGetError();
    
    std::cout << "GLEW initialized successfully" << std::endl;
    std::cout << "OpenGL version: " << glGetString(GL_VERSION) << std::endl;
    std::cout << "OpenGL renderer: " << glGetString(GL_RENDERER) << std::endl;
    
    return true;
}

bool VisualizationManager::createShaders() {
    // Create wireframe shader program
    wireframeShaderProgram = createShaderProgram(vertexShaderSource, fragmentShaderSource);
    if (!wireframeShaderProgram) {
        return false;
    }
    
    return true;
}

unsigned int VisualizationManager::compileShader(GLenum type, const char* source) {
    // Create shader
    unsigned int shader = glCreateShader(type);
    glShaderSource(shader, 1, &source, nullptr);
    glCompileShader(shader);
    
    // Check compilation status
    int success;
    char infoLog[512];
    glGetShaderiv(shader, GL_COMPILE_STATUS, &success);
    if (!success) {
        glGetShaderInfoLog(shader, sizeof(infoLog), nullptr, infoLog);
        std::cerr << "Shader compilation error: " << infoLog << std::endl;
        glDeleteShader(shader);
        return 0;
    }
    
    return shader;
}

unsigned int VisualizationManager::createShaderProgram(const char* vertexSource, const char* fragmentSource) {
    // Compile vertex shader
    unsigned int vertexShader = compileShader(GL_VERTEX_SHADER, vertexSource);
    if (!vertexShader) {
        return 0;
    }
    
    // Compile fragment shader
    unsigned int fragmentShader = compileShader(GL_FRAGMENT_SHADER, fragmentSource);
    if (!fragmentShader) {
        glDeleteShader(vertexShader);
        return 0;
    }
    
    // Create shader program
    unsigned int program = glCreateProgram();
    glAttachShader(program, vertexShader);
    glAttachShader(program, fragmentShader);
    glLinkProgram(program);
    
    // Check linking status
    int success;
    char infoLog[512];
    glGetProgramiv(program, GL_LINK_STATUS, &success);
    if (!success) {
        glGetProgramInfoLog(program, sizeof(infoLog), nullptr, infoLog);
        std::cerr << "Shader program linking error: " << infoLog << std::endl;
        glDeleteShader(vertexShader);
        glDeleteShader(fragmentShader);
        glDeleteProgram(program);
        return 0;
    }
    
    // Clean up shaders (they're linked to the program now)
    glDeleteShader(vertexShader);
    glDeleteShader(fragmentShader);
    
    return program;
}

//-----------------------------------------------------------------------------
// Frame Management
//-----------------------------------------------------------------------------

void VisualizationManager::beginFrame() {
    // Clear the color and depth buffers
    glClearColor(0.1f, 0.1f, 0.1f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
}

void VisualizationManager::endFrame() {
    // Swap buffers
    glfwSwapBuffers(window);
    
    // Poll for events
    glfwPollEvents();
}

//-----------------------------------------------------------------------------
// Component Rendering
//-----------------------------------------------------------------------------

void VisualizationManager::renderComponents(SimulationManager& simManager) {
    // Get all components from the simulation manager
    const auto& components = simManager.getComponents();
    
    // Render each visualizable component
    for (const auto& component : components) {
        if (component && component->isVisualizable()) {
            // Initialize visualization for this component if needed
            initializeComponentVisualization(component);
            
            // Let the component prepare its visualization data
            component->prepareForVisualization();
            
            // Let the component visualize itself using this manager
            component->visualize(*this);
        }
    }
}

void VisualizationManager::initializeComponentVisualization(std::shared_ptr<ComponentInterface> component) {
    if (component && component->isVisualizable()) {
        // Call the component's visualization initialization method
        component->initializeVisualization(*this);
    }
}

//-----------------------------------------------------------------------------
// Resource Creation Methods
//-----------------------------------------------------------------------------

unsigned int VisualizationManager::createVertexBuffer(size_t size, const void* data) {
    unsigned int vbo;
    glGenBuffers(1, &vbo);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, size, data, GL_DYNAMIC_DRAW);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    return vbo;
}

unsigned int VisualizationManager::createVertexArray() {
    unsigned int vao;
    glGenVertexArrays(1, &vao);
    return vao;
}

unsigned int VisualizationManager::createIndexBuffer(const void* data, size_t size) {
    unsigned int ebo;
    glGenBuffers(1, &ebo);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, size, data, GL_STATIC_DRAW);
    return ebo;
}

void VisualizationManager::configureVertexAttributes(unsigned int vao, unsigned int vbo, unsigned int location, 
                                                    int size, size_t stride, size_t offset) {
    glBindVertexArray(vao);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glEnableVertexAttribArray(location);
    glVertexAttribPointer(location, size, GL_FLOAT, GL_FALSE, stride, reinterpret_cast<void*>(offset));
    glBindVertexArray(0);
}

//-----------------------------------------------------------------------------
// Rendering Methods
//-----------------------------------------------------------------------------

void VisualizationManager::renderWireframe(unsigned int vao, unsigned int ebo, int indexCount, const glm::vec3& color) {
    // Use wireframe shader
    glUseProgram(wireframeShaderProgram);
    
    // Set wireframe mode
    glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
    
    // Calculate model-view-projection matrices
    glm::mat4 model = glm::mat4(1.0f);
    glm::mat4 view = camera->getViewMatrix();
    glm::mat4 projection = camera->getProjectionMatrix();
    
    // Set uniforms
    glUniformMatrix4fv(glGetUniformLocation(wireframeShaderProgram, "model"), 1, GL_FALSE, glm::value_ptr(model));
    glUniformMatrix4fv(glGetUniformLocation(wireframeShaderProgram, "view"), 1, GL_FALSE, glm::value_ptr(view));
    glUniformMatrix4fv(glGetUniformLocation(wireframeShaderProgram, "projection"), 1, GL_FALSE, glm::value_ptr(projection));
    glUniform3fv(glGetUniformLocation(wireframeShaderProgram, "color"), 1, glm::value_ptr(color));
    
    // Bind VAO and EBO
    glBindVertexArray(vao);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo);
    
    // Draw wireframe
    glDrawElements(GL_LINES, indexCount, GL_UNSIGNED_INT, 0);
    
    // Reset to fill mode
    glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
    
    // Unbind VAO
    glBindVertexArray(0);
}

//-----------------------------------------------------------------------------
// Window Status
//-----------------------------------------------------------------------------

bool VisualizationManager::shouldClose() const {
    return glfwWindowShouldClose(window);
}

//-----------------------------------------------------------------------------
// Cleanup
//-----------------------------------------------------------------------------

void VisualizationManager::shutdown() {
    // Delete shaders
    if (wireframeShaderProgram) {
        glDeleteProgram(wireframeShaderProgram);
        wireframeShaderProgram = 0;
    }
    
    // Destroy window
    if (window) {
        glfwDestroyWindow(window);
        window = nullptr;
    }
    
    // Terminate GLFW
    glfwTerminate();
    
    std::cout << "VisualizationManager shutdown complete" << std::endl;
}

} // namespace drumforge