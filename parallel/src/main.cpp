#include "cuda_memory_manager.h"
#include "simulation_manager.h"
#include "membrane_component.h"
#include "body_component.h"
#include "visualization_manager.h"
#include "input_handler.h"
#include "gui_manager.h"
#include <iostream>
#include <memory>
#include <chrono>
#include <thread>
#include <stdexcept>
#include <string>

// Global flag to control CUDA-OpenGL interop attempts
bool g_enableCudaGLInterop = false;  // Set to false to disable interop completely

// Debug function to check for CUDA errors
void checkCudaErrorsDebug(const char* filename, int line) {
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        std::cerr << "CUDA Error at " << filename << ":" << line 
                  << ": " << cudaGetErrorString(error) << std::endl;
        throw std::runtime_error("CUDA Error detected");
    }
}

#define CHECK_CUDA_ERRORS() checkCudaErrorsDebug(__FILE__, __LINE__)

int main(int argc, char* argv[]) {
    std::cout << "DrumForge Parallel - Starting up..." << std::endl;
    
    // Parse command line arguments
    if (argc > 1 && std::string(argv[1]) == "--enable-interop") {
        g_enableCudaGLInterop = true;
        std::cout << "CUDA-OpenGL interoperability enabled via command line flag" << std::endl;
    } else {
        std::cout << "CUDA-OpenGL interoperability disabled (use --enable-interop to enable)" << std::endl;
    }
    
    try {
        // IMPORTANT: Initialize VISUALIZATION MANAGER FIRST!
        // This is critical for proper CUDA-OpenGL interop - OpenGL context must exist
        // before CUDA tries to use it
        std::cout << "Initializing visualization manager..." << std::endl;
        drumforge::VisualizationManager& visManager = drumforge::VisualizationManager::getInstance();
        if (!visManager.initialize(1280, 720, "DrumForge")) {
            std::cerr << "Failed to initialize visualization - cannot continue" << std::endl;
            return 1;
        }
        std::cout << "Visualization initialized successfully" << std::endl;
        
        // After OpenGL is initialized, then initialize CUDA
        std::cout << "Initializing CUDA memory manager..." << std::endl;
        drumforge::CudaMemoryManager& cudaManager = drumforge::CudaMemoryManager::getInstance();
        
        cudaManager.initialize();
        CHECK_CUDA_ERRORS();
        
        // Check if interop is supported only if we want to use it
        if (g_enableCudaGLInterop) {
            bool interopSupported = cudaManager.isGLInteropSupported();
            if (!interopSupported) {
                std::cerr << "WARNING: CUDA-OpenGL interop is not supported on this system. "
                        << "Falling back to CPU-based visualization." << std::endl;
                g_enableCudaGLInterop = false;  // Disable interop for the rest of the program
            }
        }

        // After other manager initializations:
        drumforge::AudioManager& audioManager = drumforge::AudioManager::getInstance();
        audioManager.initialize(44100);  // 44.1kHz sample rate
        
        // Get the simulation manager
        drumforge::SimulationManager& simManager = drumforge::SimulationManager::getInstance();
        
        // Initialize GUI manager
        std::cout << "Initializing GUI manager..." << std::endl;
        drumforge::GUIManager& guiManager = drumforge::GUIManager::getInstance();
        guiManager.initialize(visManager.getWindow());
        
        // Create a shared pointer for the membrane component (will initialize later)
        std::shared_ptr<drumforge::MembraneComponent> membrane = nullptr;
        
        // Fixed timestep for simulation
        const float timestep = 1.0f / 1.1f;  // ~60 FPS
        bool simulationInitialized = false;
        
        // Main loop
        while (!visManager.shouldClose()) {
            // Process input (camera movement, mouse clicks, etc.)
            auto inputHandler = visManager.getInputHandler();
            if (inputHandler) {
                float realDeltaTime = timestep;  // Use wall-clock time for camera controls
                inputHandler->processInput(realDeltaTime);
                
                // Check for escape key to close the window
                if (inputHandler->shouldClose()) {
                    break;
                }
            }
            
            // Begin GUI frame
            guiManager.beginFrame();
            
            // Check current application state
            drumforge::AppState currentState = guiManager.getState();
            
            // Check if we need to initialize the simulation for the first time
            if (guiManager.shouldInitializeSimulation() && !simulationInitialized) {
                // Get configuration parameters from GUI
                int gridSizeX = guiManager.getConfigGridSizeX();
                int gridSizeY = guiManager.getConfigGridSizeY();
                int gridSizeZ = guiManager.getConfigGridSizeZ();
                float cellSize = guiManager.getConfigCellSize();
                float membraneRadius = guiManager.getConfigRadius();
                float membraneTension = guiManager.getConfigTension();
                float membraneDamping = guiManager.getConfigDamping();
                float timeScale = guiManager.getConfigTimeScale();
                
                // Initialize the simulation
                simManager.initialize();
                CHECK_CUDA_ERRORS();
                
                std::cout << "SimulationManager initialized successfully!" << std::endl;
                
                // Apply configuration parameters
                drumforge::SimulationParameters params = simManager.getParameters();
                params.timeScale = timeScale;
                params.gridSizeX = gridSizeX;
                params.gridSizeY = gridSizeY;
                params.gridSizeZ = gridSizeZ;
                params.cellSize = cellSize;
                simManager.updateParameters(params);
                
                std::cout << "Simulation parameters updated (gridSize = " 
                          << params.gridSizeX << "x" << params.gridSizeY << "x" << params.gridSizeZ
                          << ", cellSize = " << params.cellSize << ")" << std::endl;
                
                // Create the membrane component
                std::cout << "Creating membrane component..." << std::endl;
                membrane = std::make_shared<drumforge::MembraneComponent>(
                    "Drumhead", 
                    membraneRadius,   // Radius
                    membraneTension,  // Tension
                    membraneDamping   // Damping
                );
                
                // Add membrane to simulation
                simManager.addComponent(membrane);
                
                // Initialize the membrane component
                membrane->initialize();
                CHECK_CUDA_ERRORS();
                std::cout << "Membrane component initialized successfully" << std::endl;
                
                // Connect the membrane to the input handler for click interaction
                if (inputHandler) {
                    inputHandler->connectMembrane(membrane);
                    std::cout << "Membrane connected to input handler - click on the membrane to apply impulses" << std::endl;
                }

                // Create the body component with the same radius as the membrane
                std::cout << "Creating body component..." << std::endl;
                std::shared_ptr<drumforge::BodyComponent> body = std::make_shared<drumforge::BodyComponent>(
                    "DrumShell",
                    membraneRadius,             // Same radius as the membrane
                    membraneRadius * 0.8f,      // Height (80% of radius for reasonable proportions)
                    membraneRadius * 0.05f,     // Thickness (5% of radius)
                    "Maple"                     // Default material
                );

                // Add body to simulation
                simManager.addComponent(body);

                // Set custom parameters before initialization
                // This will reduce the number of modes and limit the max frequency
                // We'll add this code right before calling initialize()
                body->setCustomParameters(32, 60.0f, 1200.0f);

                // Initialize the body component
                body->initialize();
                CHECK_CUDA_ERRORS();
                std::cout << "Body component initialized successfully" << std::endl;
                
                // Mark simulation as initialized
                simulationInitialized = true;
                guiManager.setSimulationInitialized();
            }
            
            // Only advance simulation if we're in simulation state and it's initialized
            if (currentState == drumforge::AppState::SIMULATION && simulationInitialized) {
                // Advance simulation
                simManager.advance(timestep);
                CHECK_CUDA_ERRORS();
            }
            
            // Begin rendering frame
            visManager.beginFrame();
            
            // Only render simulation components if in simulation state and initialized
            if (currentState == drumforge::AppState::SIMULATION && simulationInitialized) {
                visManager.renderComponents(simManager);
            }
            
            // Render GUI (this handles rendering the appropriate UI based on state)
            guiManager.renderGUI(simManager, membrane);
            guiManager.renderFrame();
            
            // Finish frame
            visManager.endFrame();
            CHECK_CUDA_ERRORS();
            
            // Clear state changed flag if it was set
            if (guiManager.hasStateChanged()) {
                guiManager.clearStateChanged();
            }
            
            // Limit frame rate 
            std::this_thread::sleep_for(std::chrono::milliseconds(16)); // ~60 FPS
        }
        
        std::cout << "Visualization test complete" << std::endl;
        
        // Clean up - order is important for clean shutdown
        // First, clean up GUI resources
        guiManager.shutdown();
        
        // Then clean up visualization resources
        visManager.shutdown();
        
        // Then shutdown simulation
        simManager.shutdown();
        
        // Finally, clean up CUDA resources
        cudaManager.shutdown();
        
        std::cout << "DrumForge Parallel - Shutdown complete" << std::endl;
    }
    catch (const drumforge::CudaException& e) {
        std::cerr << "CUDA Error: " << e.what() << std::endl;
        return 1;
    }
    catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    catch (...) {
        std::cerr << "Unknown error occurred" << std::endl;
        return 1;
    }

    return 0;
}