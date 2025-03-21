#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <iostream>
#include <vector>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <GL/glu.h>
#include "membrane.h"
#include "shader.h"
#include "airspace.h"
#include "camera.h"  // Include the new camera header

// Global variables
unsigned int VAO, VBO;
Shader* pointShader;
DrumMembrane membrane(32, 10.0f, 10.0f, 0.1f);
// Change from pointer to object
AirSpace airSpace(32, 32, 16, 1.0f); 
const float impulseStrength = 0.1f;
float lastFrameTime = 0.0f;

// Mouse interaction variables
bool mousePressed = false;

// Camera object
Camera* camera = nullptr;

// Display controls
bool showAirspace = true;
bool showMembrane = true;
bool showGrid = true;

// Print keyboard controls
void printControls() {
    std::cout << "\n=== Keyboard Controls ===\n";
    std::cout << "Camera Movement:\n";
    std::cout << "  W/S - Move forward/backward\n";
    std::cout << "  A/D - Move left/right\n";
    std::cout << "  Q/E - Move up/down\n";
    
    std::cout << "\nCamera Panning:\n";
    std::cout << "  Arrow Keys - Pan camera view\n";
    
    std::cout << "\nVisibility Toggles:\n";
    std::cout << "  1 - Toggle airspace visibility\n";
    std::cout << "  2 - Toggle membrane visibility\n";
    std::cout << "  3 - Toggle grid visibility\n";
    
    std::cout << "\nOther Controls:\n";
    std::cout << "  R - Reset membrane to flat state\n";
    std::cout << "  P - Add pressure impulse to airspace\n";
    std::cout << "  H - Show this help message\n";
    std::cout << "  ESC - Exit application\n";
    std::cout << "======================\n\n";
}

void addTestImpulse() {
    if (!showAirspace) return;
    
    // Add a pressure impulse in the center of the airspace
    float centerX = airSpace.getSizeX() / 2.0f;
    float centerY = airSpace.getSizeY() / 2.0f;
    float centerZ = airSpace.getSizeZ() / 2.0f;
    
    airSpace.addPressureImpulse(centerX, centerY, centerZ, 10.0f, 5.0f);
    std::cout << "Added pressure impulse at center: (" << centerX << ", " << centerY << ", " << centerZ << ")" << std::endl;
}

// Handle key press events (called by GLFW)
void key_callback(GLFWwindow* window, int key, int scancode, int action, int mods) {
    // Only handle key press events (not repeat or release)
    if (action != GLFW_PRESS) return;
    
    // Handle various key commands using a switch statement for easy expansion
    switch (key) {
        // Visibility toggles
        case GLFW_KEY_1:
            showAirspace = !showAirspace;
            std::cout << "Airspace visibility: " << (showAirspace ? "ON" : "OFF") << std::endl;
            break;
            
        case GLFW_KEY_2:
            showMembrane = !showMembrane;
            std::cout << "Membrane visibility: " << (showMembrane ? "ON" : "OFF") << std::endl;
            break;
            
        case GLFW_KEY_3:
            showGrid = !showGrid;
            std::cout << "Grid visibility: " << (showGrid ? "ON" : "OFF") << std::endl;
            break;
            
        // Reset membrane
        case GLFW_KEY_R:
            membrane.reset();
            std::cout << "Membrane reset to flat state" << std::endl;
            break;
            
        // Show help
        case GLFW_KEY_H:
            printControls();
            break;
            
        // Exit application
        case GLFW_KEY_ESCAPE:
            glfwSetWindowShouldClose(window, GLFW_TRUE);
            break;

        case GLFW_KEY_P:  // Press 'P' to add a pressure impulse
            addTestImpulse();
            break;
    }
}

// Mouse button callback function
void mouse_button_callback(GLFWwindow* window, int button, int action, int mods) {
    if (button == GLFW_MOUSE_BUTTON_LEFT && action == GLFW_PRESS) {
        mousePressed = true;
        
        // Get cursor position
        double xpos, ypos;
        glfwGetCursorPos(window, &xpos, &ypos);
        
        // Get window size
        int width, height;
        glfwGetWindowSize(window, &width, &height);
        
        // Convert to normalized device coordinates (-1 to 1)
        float ndcX = (2.0f * xpos) / width - 1.0f;
        float ndcY = 1.0f - (2.0f * ypos) / height;
        
        // Set up matrices (these should match your rendering matrices)
        glm::mat4 projection = glm::perspective(glm::radians(45.0f), 1.0f, 0.1f, 1000.0f);
        
        // Use current camera position and view matrix
        glm::mat4 view = camera->getViewMatrix();
        
        // Create a ray from camera through clicked point
        glm::vec4 rayClip = glm::vec4(ndcX, ndcY, -1.0, 1.0);
        glm::mat4 invProjection = glm::inverse(projection);
        glm::vec4 rayEye = invProjection * rayClip;
        rayEye = glm::vec4(rayEye.x, rayEye.y, -1.0, 0.0);
        
        glm::mat4 invView = glm::inverse(view);
        glm::vec4 rayWorld = invView * rayEye;
        glm::vec3 rayDirection = glm::normalize(glm::vec3(rayWorld));
        
        // Origin of the ray (camera position)
        glm::vec3 rayOrigin = camera->getPosition();
        
        // Intersect ray with the membrane plane (z = 0)
        // P = O + t*D, where t = -(O.z) / D.z
        float t = -rayOrigin.z / rayDirection.z;
        glm::vec3 intersectionPoint = rayOrigin + t * rayDirection;
        
        // Convert intersection point to grid coordinates
        float gridX = intersectionPoint.x;
        float gridY = intersectionPoint.y;
        
        // Normalize to [0,1] for the membrane
        float normalizedX = gridX / membrane.getGridSize();
        float normalizedY = gridY / membrane.getGridSize();
        
        // Apply impulse at the calculated position
        membrane.applyImpulse(normalizedX, normalizedY, impulseStrength);
        
        std::cout << "Applied impulse at grid: " << gridX << ", " << gridY 
                  << " (normalized: " << normalizedX << ", " << normalizedY << ")" << std::endl;
    } else if (button == GLFW_MOUSE_BUTTON_LEFT && action == GLFW_RELEASE) {
        mousePressed = false;
    }
}

// Draw AirSpace as a wireframe cube
void drawAirSpace() {
    if (!showAirspace) return;
    
    // Get dimensions
    float minX = airSpace.getPosX();
    float minY = airSpace.getPosY();
    float minZ = airSpace.getPosZ();
    float maxX = minX + airSpace.getSizeX() * airSpace.getCellSize();
    float maxY = minY + airSpace.getSizeY() * airSpace.getCellSize();
    float maxZ = minZ + airSpace.getSizeZ() * airSpace.getCellSize();
    
    // Set cube color to red
    glColor3f(1.0f, 0.0f, 0.0f);
    glLineWidth(2.0f);  // Thicker lines for visibility
    
    // Draw wireframe cube
    glBegin(GL_LINES);
    
    // Bottom face
    glVertex3f(minX, minY, minZ); glVertex3f(maxX, minY, minZ);
    glVertex3f(maxX, minY, minZ); glVertex3f(maxX, maxY, minZ);
    glVertex3f(maxX, maxY, minZ); glVertex3f(minX, maxY, minZ);
    glVertex3f(minX, maxY, minZ); glVertex3f(minX, minY, minZ);
    
    // Top face
    glVertex3f(minX, minY, maxZ); glVertex3f(maxX, minY, maxZ);
    glVertex3f(maxX, minY, maxZ); glVertex3f(maxX, maxY, maxZ);
    glVertex3f(maxX, maxY, maxZ); glVertex3f(minX, maxY, maxZ);
    glVertex3f(minX, maxY, maxZ); glVertex3f(minX, minY, maxZ);
    
    // Connecting lines
    glVertex3f(minX, minY, minZ); glVertex3f(minX, minY, maxZ);
    glVertex3f(maxX, minY, minZ); glVertex3f(maxX, minY, maxZ);
    glVertex3f(maxX, maxY, minZ); glVertex3f(maxX, maxY, maxZ);
    glVertex3f(minX, maxY, minZ); glVertex3f(minX, maxY, maxZ);
    
    glEnd();
    
    // Reset line width
    glLineWidth(1.0f);
}

// 2. Modify the drawAirSpacePressure function to use the public method:
void drawAirSpacePressure() {
    if (!showAirspace) return;
    
    // Get pressure field for visualization
    const std::vector<float>& pressureField = airSpace.getPressureField();
    
    // Draw a slice at some fixed Y value to see the pressure waves
    int sliceY = airSpace.getSizeY() / 2;
    
    glPointSize(3.0f);
    glBegin(GL_POINTS);
    
    for (int z = 0; z < airSpace.getSizeZ(); z++) {
        for (int x = 0; x < airSpace.getSizeX(); x++) {
            // Get pressure at this point using the new public method
            float pressure = airSpace.getPressureAtGrid(x, sliceY, z);
            
            // Map pressure to color (blue for negative, white for zero, red for positive)
            float red = pressure > 0 ? pressure / 10.0f : 0.0f;
            float blue = pressure < 0 ? -pressure / 10.0f : 0.0f;
            float green = 0.0f;
            
            // Clamp colors to [0,1]
            red = std::min(1.0f, std::max(0.0f, red));
            blue = std::min(1.0f, std::max(0.0f, blue));
            
            // Draw the point
            glColor3f(red, green, blue);
            glVertex3f(x, sliceY, z);
        }
    }
    
    glEnd();
}

// Draw XYZ axes at origin
void drawAxes() {
    glLineWidth(2.0f);
    glBegin(GL_LINES);
    
    // X axis (red)
    glColor3f(1.0f, 0.0f, 0.0f);
    glVertex3f(0.0f, 0.0f, 0.0f);
    glVertex3f(10.0f, 0.0f, 0.0f);
    
    // Y axis (green)
    glColor3f(0.0f, 1.0f, 0.0f);
    glVertex3f(0.0f, 0.0f, 0.0f);
    glVertex3f(0.0f, 10.0f, 0.0f);
    
    // Z axis (blue)
    glColor3f(0.0f, 0.0f, 1.0f);
    glVertex3f(0.0f, 0.0f, 0.0f);
    glVertex3f(0.0f, 0.0f, 10.0f);
    
    glEnd();
    glLineWidth(1.0f);
}

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

// Cleanup function
void cleanupGL() {
    glDeleteBuffers(1, &VBO);
    glDeleteVertexArrays(1, &VAO);
    delete pointShader;
    delete camera;  // Clean up camera
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
    
    // Set callbacks
    glfwSetMouseButtonCallback(window, mouse_button_callback);
    glfwSetKeyCallback(window, key_callback);
    
    // Enable VSync
    glfwSwapInterval(1);
    
    std::cout << "Window created successfully." << std::endl;

    // Initialize OpenGL
    initializeGL();
    
    // Setup membrane and initial buffer data
    updateVertexData();
    
    // Position the airspace to align with the membrane
    airSpace.setPosition(0.0f, 0.0f, 0.0f);
    
    // Set up initial camera position
    float gridCenter = membrane.getGridSize() / 2.0f;
    float gridSize = membrane.getGridSize();
    
    // Initialize camera with position and target
    glm::vec3 initialPos = glm::vec3(gridCenter - gridSize * 0.8f, gridCenter - gridSize * 0.5f, gridSize * 0.9f);
    glm::vec3 initialTarget = glm::vec3(gridCenter, gridCenter, 0.0f);
    camera = new Camera(initialPos, initialTarget);
    
    // Print controls
    printControls();

    // Main rendering loop
    while (!glfwWindowShouldClose(window)) {
        // Calculate delta time
        float currentTime = glfwGetTime();
        float deltaTime = currentTime - lastFrameTime;
        lastFrameTime = currentTime;
        
        // Process camera movement and panning with our new Camera class
        camera->processMovement(window, deltaTime);
        camera->processPanning(window, deltaTime);
        
        static float accumulator = 0.0f;

        // Global variables
        float simulationSpeed = 10.0f;  // How much faster than real-time to run
        const float fixedPhysicsTimestep = 1.0f / 5.0f;  // Keep this constant for consistent physics

        // Update simulation with fixed time step for stability but apply simulation speed
        accumulator += deltaTime * simulationSpeed;

        // Run physics updates with the SAME fixed timestep
        while (accumulator >= fixedPhysicsTimestep) {
            membrane.updateSimulation(fixedPhysicsTimestep);
            airSpace.updateSimulation(fixedPhysicsTimestep);  // Now with timestep parameter
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
        
        // Use a perspective projection with extended far plane for better visibility
        float aspectRatio = static_cast<float>(windowWidth) / windowHeight;
        gluPerspective(45.0f, aspectRatio, 0.1f, 1000.0f);
        
        glMatrixMode(GL_MODELVIEW);
        glLoadIdentity();
        
        // Use camera's view matrix
        glm::mat4 viewMatrix = camera->getViewMatrix();
        glMultMatrixf(glm::value_ptr(viewMatrix));
        
        // Draw coordinate axes for reference
        drawAxes();
        
        // Draw the airspace wireframe
        drawAirSpace();
        
        if (showGrid) {
            // Draw the grid lines
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
        }
        
        if (showMembrane) {
            // Draw the membrane points
            glPointSize(3.0f);
            glBegin(GL_POINTS);
            for (int y = 0; y < membrane.getGridSize(); y++) {
                for (int x = 0; x < membrane.getGridSize(); x++) {
                    if (membrane.isInsideCircle(x, y)) {
                        // Get the current height from the simulation
                        float height = membrane.getHeight(x, y);
                        
                        // Scale the height for better visibility
                        float scaledHeight = height * 5.0f;
                                            
                        // Use a color gradient based on height
                        float r = 0.0f;
                        float g = 0.6f + scaledHeight * 0.4f;  // More green for positive heights
                        float b = 1.0f - fabs(scaledHeight) * 0.5f;  // Less blue for extreme heights
                        glColor3f(r, g, b);
                        
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
        }

        drawAirSpacePressure();
        
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