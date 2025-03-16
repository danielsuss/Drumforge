#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <iostream>
#include <vector>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include "membrane.h"
#include "shader.h"

unsigned int VAO, VBO;
Shader* pointShader;
DrumMembrane membrane(512, 100.0f);

void initializeGL() {
    // Initialize GLEW
    glewInit();

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

    std::cout << "Generated " << vertices.size() << " vertices" << std::endl;

    // Print the first few vertices to check they look correct
    if (!vertices.empty()) {
        std::cout << "First vertex: (" << vertices[0].x << ", " << vertices[0].y << ", " << vertices[0].z << ")" << std::endl;
        int midIndex = vertices.size() / 2;
        std::cout << "Middle vertex: (" << vertices[midIndex].x << ", " << vertices[midIndex].y << ", " << vertices[midIndex].z << ")" << std::endl;
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
    glfwInit();
    
    // Set OpenGL version hints
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_COMPAT_PROFILE);
    
    // Create a window
    const int windowWidth = 800;
    const int windowHeight = 800;
    GLFWwindow* window = glfwCreateWindow(
        windowWidth, windowHeight, 
        "DrumForge", nullptr, nullptr
    );
    
    // Make the window's context current
    glfwMakeContextCurrent(window);
    
    // Enable VSync
    glfwSwapInterval(1);
    
    std::cout << "Window created successfully." << std::endl;

    // Initialize OpenGL
    initializeGL();
    
    // Setup membrane and initial buffer data
    updateVertexData();

    // Setup projection and view matrices
    float gridSize = static_cast<float>(membrane.getGridSize());
    glm::mat4 projection = glm::ortho(-10.0f, gridSize + 10.0f, -10.0f, gridSize + 10.0f, -100.0f, 100.0f);
    glm::mat4 view = glm::lookAt(
        glm::vec3(0.0f, 0.0f, 2.0f),   // Camera position (moved closer)
        glm::vec3(0.0f, 0.0f, 0.0f),   // Look at origin
        glm::vec3(0.0f, 1.0f, 0.0f)    // Up vector
    );
    
    // Main rendering loop
    while (!glfwWindowShouldClose(window)) {
        // Clear the screen
        glClear(GL_COLOR_BUFFER_BIT);
        glClearColor(0.2f, 0.2f, 0.2f, 1.0f);
        
        // Set up a simple projection matrix
        glMatrixMode(GL_PROJECTION);
        glLoadIdentity();
        glOrtho(-1.0, 1.0, -1.0, 1.0, -1.0, 1.0);
        
        // Set up modelview matrix
        glMatrixMode(GL_MODELVIEW);
        glLoadIdentity();
        
        // // Draw a simple triangle
        // glBegin(GL_TRIANGLES);
        // glColor3f(1.0f, 0.0f, 0.0f); // Red
        // glVertex3f(-0.5f, -0.5f, 0.0f);
        // glColor3f(0.0f, 1.0f, 0.0f); // Green  
        // glVertex3f(0.5f, -0.5f, 0.0f);
        // glColor3f(0.0f, 0.0f, 1.0f); // Blue
        // glVertex3f(0.0f, 0.5f, 0.0f);
        // glEnd();

        // Draw the membrane points using the fixed function pipeline
        glPointSize(1.0f); // Make points visible

        // Set up the matrices for the membrane visualization
        glMatrixMode(GL_PROJECTION);
        glLoadIdentity();
        // Use an orthographic projection that matches the grid size
        glOrtho(0, membrane.getGridSize(), 0, membrane.getGridSize(), -10, 10);

        glMatrixMode(GL_MODELVIEW);
        glLoadIdentity();

        // Draw only points inside the circle
        int gridSize = membrane.getGridSize();
        glBegin(GL_POINTS);
        for (int y = 0; y < gridSize; y++) {
            for (int x = 0; x < gridSize; x++) {
                if (membrane.isInsideCircle(x, y)) {
                    // Blue for points inside membrane
                    glColor3f(0.0f, 0.6f, 1.0f);
                } else {
                    // Gray for points outside membrane
                    glColor3f(0.3f, 0.3f, 0.3f);
                }
                glVertex3f(x, y, 0.0f);
            }
        }
        glEnd();
        
        // Swap buffers and poll events
        glfwSwapBuffers(window);
        glfwPollEvents();
    }
}