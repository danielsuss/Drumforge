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
DrumMembrane membrane(128, 1.0f);

void initializeGL() {
    // Initialize GLEW
    glewInit();
    
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
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    
    // Create a window
    const int windowWidth = 800;
    const int windowHeight = 600;
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
    glm::mat4 projection = glm::ortho(-1.2f, 1.2f, -1.2f, 1.2f, -1.0f, 100.0f);
    glm::mat4 view = glm::lookAt(
        glm::vec3(0.0f, 0.0f, 3.0f),   // Camera position
        glm::vec3(0.0f, 0.0f, 0.0f),   // Look at origin
        glm::vec3(0.0f, 1.0f, 0.0f)    // Up vector
    );
    
    // Main rendering loop
    while (!glfwWindowShouldClose(window)) {
        // Clear the screen
        glClear(GL_COLOR_BUFFER_BIT);
        
        // Set background color (dark gray)
        glClearColor(0.2f, 0.2f, 0.2f, 1.0f);

         // Use shader
        pointShader->use();
        
        // Set uniforms
        pointShader->setMat4("projection", projection);
        pointShader->setMat4("view", view);

        // Draw membrane points
        glBindVertexArray(VAO);
        int gridSize = membrane.getGridSize();
        for (int i = 0; i < gridSize * gridSize; i++) {
            int x = i % gridSize;
            int y = i / gridSize;
            
            // Set point color based on whether it's inside the membrane
            if (membrane.isInsideCircle(x, y)) {
                // Blue for points inside membrane
                pointShader->setVec4("pointColor", glm::vec4(0.0f, 0.6f, 1.0f, 1.0f));
            } else {
                // Gray for points outside membrane
                pointShader->setVec4("pointColor", glm::vec4(0.3f, 0.3f, 0.3f, 1.0f));
            }
            
            // Draw a single point
            glDrawArrays(GL_POINTS, i, 1);
        }
        glBindVertexArray(0);
        
        // Swap buffers and poll events
        glfwSwapBuffers(window);
        glfwPollEvents();
    }
    
    // Clean up
    glfwDestroyWindow(window);
    glfwTerminate();
    
    return 0;
}