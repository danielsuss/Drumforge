#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <iostream>
#include <vector>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <GL/glu.h>  // Add GLU for gluPerspective and gluLookAt
#include "membrane.h"
#include "shader.h"

// Global variables
unsigned int VAO, VBO;
Shader* pointShader;
// Create membrane with appropriate parameters:
// Size: 32x32 grid points
// Radius: 10.0 units
// Tension: 1.0 (default)
// Damping: 0.001 (very small damping to allow oscillation with slow decay)
DrumMembrane membrane(32, 10.0f, 1.0f, 0.01f);
float lastFrameTime = 0.0f;

// Mouse interaction variables
bool mousePressed = false;

void initializeGL() {
    // Initialize GLEW
    GLenum err = glewInit();
    if (err != GLEW_OK) {
        std::cerr << "GLEW initialization failed: " << glewGetErrorString(err) << std::endl;
        return;
    }
    std::cout << "GLEW Version: " << glewGetString(GLEW_VERSION) << std::endl;
    std::cout << "OpenGL Version: " << glGetString(GL_VERSION) << std::endl;
    std::cout << "GLSL Version: " << glGetString(GL_SHADING_LANGUAGE_VERSION) << std::endl;
    std::cout << "Vendor: " << glGetString(GL_VENDOR) << std::endl;
    std::cout << "Renderer: " << glGetString(GL_RENDERER) << std::endl;
    
    // Create shader
    pointShader = new Shader("shaders/point_vertex.glsl", "shaders/point_fragment.glsl");
    
    // Generate buffers
    glGenVertexArrays(1, &VAO);
    glGenBuffers(1, &VBO);
    
    // Configure buffers
    glBindVertexArray(VAO);
    glBindBuffer(GL_ARRAY_BUFFER, VBO);
    
    // Setup vertex attributes
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(glm::vec3), (void*)0);
    glEnableVertexAttribArray(0);
    
    // Unbind
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindVertexArray(0);
    
    // Enable point size control
    glEnable(GL_PROGRAM_POINT_SIZE);
}

void updateVertexData() {
    // Generate vertices from the membrane
    std::vector<glm::vec3> vertices = membrane.generateVertices();
    
    // Update the buffer data
    glBindBuffer(GL_ARRAY_BUFFER, VBO);
    glBufferData(GL_ARRAY_BUFFER, vertices.size() * sizeof(glm::vec3), vertices.data(), GL_STATIC_DRAW);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
}

// Mouse button callback function
void mouse_button_callback(GLFWwindow* window, int button, int action, int mods) {
    if (button == GLFW_MOUSE_BUTTON_LEFT) {
        if (action == GLFW_PRESS) {
            mousePressed = true;
            
            // Get cursor position
            double xpos, ypos;
            glfwGetCursorPos(window, &xpos, &ypos);
            
            // Get window size for normalization
            int width, height;
            glfwGetWindowSize(window, &width, &height);
            
            // Normalize coordinates to [0,1] range
            float normalizedX = static_cast<float>(xpos) / width;
            // Invert Y (OpenGL Y is bottom-to-top)
            float normalizedY = 1.0f - static_cast<float>(ypos) / height;
            
            // Apply impulse at click position with strength 10.0
            membrane.applyImpulse(normalizedX, normalizedY, 0.05f);
            
            std::cout << "Applied impulse at: " << normalizedX << ", " << normalizedY << std::endl;
        } else if (action == GLFW_RELEASE) {
            mousePressed = false;
        }
    }
}

// Cleanup function
void cleanupGL() {
    glDeleteBuffers(1, &VBO);
    glDeleteVertexArrays(1, &VAO);
    delete pointShader;
}

int main() {
    std::cout << "DrumForge initializing..." << std::endl;
    
    // Initialize GLFW
    if (!glfwInit()) {
        std::cerr << "Failed to initialize GLFW" << std::endl;
        return -1;
    }
    
    // Set OpenGL version hints
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_COMPAT_PROFILE);
    glfwWindowHint(GLFW_SAMPLES, 4); // Enable 4x multisampling for smoother edges
    
    // Create a window
    const int windowWidth = 800;
    const int windowHeight = 800;
    GLFWwindow* window = glfwCreateWindow(
        windowWidth, windowHeight, 
        "DrumForge", nullptr, nullptr
    );
    
    if (!window) {
        std::cerr << "Failed to create GLFW window" << std::endl;
        glfwTerminate();
        return -1;
    }
    
    // Make the window's context current
    glfwMakeContextCurrent(window);
    
    // Set the mouse button callback
    glfwSetMouseButtonCallback(window, mouse_button_callback);
    
    // Enable VSync
    glfwSwapInterval(1);
    
    std::cout << "Window created successfully." << std::endl;

    // Initialize OpenGL
    initializeGL();
    
    // Setup membrane and initial buffer data
    updateVertexData();

    // Main rendering loop
    while (!glfwWindowShouldClose(window)) {
        // Calculate delta time
        float currentTime = glfwGetTime();
        float deltaTime = currentTime - lastFrameTime;
        lastFrameTime = currentTime;
        
        static float accumulator = 0.0f;

        // Global variables
        float simulationSpeed = 10.0f;  // How much faster than real-time to run
        const float fixedPhysicsTimestep = 1.0f / 5.0f;  // Keep this constant for consistent physics

        // In your main loop:
        // Update simulation with fixed time step for stability but apply simulation speed
        accumulator += deltaTime * simulationSpeed;

        // Run physics updates with the SAME fixed timestep
        while (accumulator >= fixedPhysicsTimestep) {
            membrane.updateSimulation(fixedPhysicsTimestep);
            accumulator -= fixedPhysicsTimestep;
        }
        
        // Update vertex data after physics simulation
        updateVertexData();
        
        // Clear the screen
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        glClearColor(0.2f, 0.2f, 0.2f, 1.0f);
        
        // Enable depth testing for proper 3D rendering
        glEnable(GL_DEPTH_TEST);
        
        // Set up the matrices for the membrane visualization
        glMatrixMode(GL_PROJECTION);
        glLoadIdentity();
        
        // Use a perspective projection for better 3D visualization
        // Parameters: field of view, aspect ratio, near plane, far plane
        float aspectRatio = 1.0f; // Assuming square window
        gluPerspective(45.0f, aspectRatio, 0.1f, 100.0f);
        
        glMatrixMode(GL_MODELVIEW);
        glLoadIdentity();
        
        float gridCenter = membrane.getGridSize() / 2.0f;
        float gridSize = membrane.getGridSize();

        // Position the camera to view the membrane from an angle
        // Parameters: eyeX, eyeY, eyeZ, centerX, centerY, centerZ, upX, upY, upZ
        gluLookAt(
            gridCenter - gridSize * 0.8f,  // Move camera left
            gridCenter - gridSize * 0.5f,  // Move camera down slightly
            gridSize * 0.9f,               // Keep camera above membrane
            gridCenter, gridCenter, 0.0f,  // Still looking at center
            0.0f, 0.0f, 1.0f               // Changed up vector to Z axis
        );
        
        // Draw membrane using GL_POINTS
        glPointSize(6.0f); // Make points larger and more visible
        
        // Draw the grid lines first (optional)
        glColor3f(0.3f, 0.3f, 0.3f);
        glBegin(GL_LINES);
        for (int i = 0; i <= membrane.getGridSize(); i++) {
            // Draw horizontal grid lines
            glVertex3f(0, i, 0);
            glVertex3f(membrane.getGridSize(), i, 0);
            
            // Draw vertical grid lines
            glVertex3f(i, 0, 0);
            glVertex3f(i, membrane.getGridSize(), 0);
        }
        glEnd();
        
        // Draw the membrane points
        glBegin(GL_POINTS);
        for (int y = 0; y < membrane.getGridSize(); y++) {
            for (int x = 0; x < membrane.getGridSize(); x++) {
                if (membrane.isInsideCircle(x, y)) {
                    // Get the current height from the simulation
                    float height = membrane.getHeight(x, y);
                    
                    // Scale the height for better visibility
                    float scaledHeight = height * 5.0f;
                    
                    // Use a fixed color for the membrane points
                    glColor3f(0.0f, 0.6f, 1.0f);
                    
                    // Draw the point at its 3D position with Z coordinate showing height
                    glVertex3f(x, y, scaledHeight);
                } else {
                    // Gray for points outside membrane
                    glColor3f(0.3f, 0.3f, 0.3f);
                    glVertex3f(x, y, 0.0f);
                }
            }
        }
        glEnd();
        
        // Draw connecting lines to represent the membrane surface as a wireframe mesh
        glColor3f(0.0f, 0.4f, 0.8f);
        glBegin(GL_LINES);
        for (int y = 0; y < membrane.getGridSize(); y++) {
            for (int x = 0; x < membrane.getGridSize(); x++) {
                if (membrane.isInsideCircle(x, y)) {
                    float height = membrane.getHeight(x, y) * 5.0f;
                    
                    // Connect horizontally if next point is also inside circle
                    if (x + 1 < membrane.getGridSize() && membrane.isInsideCircle(x + 1, y)) {
                        float nextHeight = membrane.getHeight(x + 1, y) * 5.0f;
                        glVertex3f(x, y, height);
                        glVertex3f(x + 1, y, nextHeight);
                    }
                    
                    // Connect vertically if next point is also inside circle
                    if (y + 1 < membrane.getGridSize() && membrane.isInsideCircle(x, y + 1)) {
                        float nextHeight = membrane.getHeight(x, y + 1) * 5.0f;
                        glVertex3f(x, y, height);
                        glVertex3f(x, y + 1, nextHeight);
                    }
                }
            }
        }
        glEnd();
        
        // Disable depth testing when done
        glDisable(GL_DEPTH_TEST);
        
        // Swap buffers and poll events
        glfwSwapBuffers(window);
        glfwPollEvents();
    }
    
    // Clean up
    cleanupGL();
    glfwTerminate();
    
    return 0;
}