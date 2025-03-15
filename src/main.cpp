#include <iostream>
#include <GLFW/glfw3.h>
#include <membrane.h>

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
    
    // Main rendering loop
    while (!glfwWindowShouldClose(window)) {
        // Clear the screen
        glClear(GL_COLOR_BUFFER_BIT);
        
        // Set background color (dark gray)
        glClearColor(0.2f, 0.2f, 0.2f, 1.0f);
        
        // Swap buffers and poll events
        glfwSwapBuffers(window);
        glfwPollEvents();
    }
    
    // Clean up
    glfwDestroyWindow(window);
    glfwTerminate();

    DrumMembrane membrane(64, 1.0f);
    std::cout << "Grid size: " << membrane.getGridSize() << std::endl;
    std::cout << "Radius: " << membrane.getRadius() << std::endl;
    
    return 0;
}