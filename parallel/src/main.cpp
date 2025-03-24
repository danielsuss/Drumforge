#include "cuda_memory_manager.h"
#include "simulation_manager.h"
#include "membrane_component.h"
#include "visualization_manager.h"
#include "input_handler.h"
#include <iostream>
#include <memory>
#include <chrono>
#include <thread>
#include <stdexcept>

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
    
    try {
        // Initialize CUDA memory manager FIRST!
        // This is important for proper CUDA-OpenGL interop
        std::cout << "Initializing CUDA memory manager..." << std::endl;
        drumforge::CudaMemoryManager::getInstance().initialize();
        CHECK_CUDA_ERRORS();
        
        // Get the simulation manager
        drumforge::SimulationManager& simManager = drumforge::SimulationManager::getInstance();
        
        // Initialize the simulation
        simManager.initialize();
        CHECK_CUDA_ERRORS();
        
        std::cout << "SimulationManager initialized successfully!" << std::endl;
        
        // Modify simulation parameters - use smaller grid for testing
        drumforge::SimulationParameters params = simManager.getParameters();
        params.timeScale = 1.0f;
        params.gridSizeX = 128;
        params.gridSizeY = 128;
        params.gridSizeZ = 16;
        params.cellSize = 0.1f;
        simManager.updateParameters(params);
        
        std::cout << "Simulation parameters updated (gridSize = " 
                  << params.gridSizeX << "x" << params.gridSizeY << "x" << params.gridSizeZ
                  << ", cellSize = " << params.cellSize << ")" << std::endl;
        
        // Create a simple membrane component
        std::cout << "Creating membrane component..." << std::endl;
        auto membrane = std::make_shared<drumforge::MembraneComponent>(
            "Drumhead", 
            3.0f,    // Radius
            100.0f,   // Tension
            0.01f    // Damping
        );
        
        // Add membrane to simulation
        simManager.addComponent(membrane);
        
        // Initialize the membrane component
        membrane->initialize();
        CHECK_CUDA_ERRORS();
        std::cout << "Membrane component initialized successfully" << std::endl;
        
        // Initialize visualization
        std::cout << "Initializing visualization manager..." << std::endl;
        drumforge::VisualizationManager& visManager = drumforge::VisualizationManager::getInstance();
        if (!visManager.initialize(1280, 720, "DrumForge - Visualization Test")) {
            std::cerr << "Failed to initialize visualization" << std::endl;
            return 1;
        }
        
        std::cout << "Visualization initialized successfully" << std::endl;
        
        // Test the visualization loop
        std::cout << "Starting visualization test loop..." << std::endl;
        
        // Fixed timestep for simulation
        const float timestep = 1.0f / 5.0f;  // ~60 FPS
        
        // Apply initial impulse to the membrane
        // membrane->applyImpulse(0.5f, 0.5f, 0.1f);
        CHECK_CUDA_ERRORS();
        
        // Main visualization test loop - run until window is closed
        int frameCount = 0;
        
        // Connect the membrane to the input handler for click interaction
        auto inputHandler = visManager.getInputHandler();
        if (inputHandler) {
            inputHandler->connectMembrane(membrane);
            std::cout << "Membrane connected to input handler - click on the membrane to apply impulses" << std::endl;
        }
        
        // Main loop
        while (!visManager.shouldClose()) {
            frameCount++;
            
            // Process input (camera movement, mouse clicks, etc.)
            if (inputHandler) {
                inputHandler->processInput(timestep);
                
                // Check for escape key to close the window
                if (inputHandler->shouldClose()) {
                    break;
                }
            }
            
            // Advance simulation
            simManager.advance(timestep);
            
            // Render frame
            visManager.beginFrame();
            visManager.renderComponents(simManager);
            visManager.endFrame();
            
            // Print status every 60 frames
            // if (frameCount % 60 == 0) {
            //     std::cout << "Frame " << frameCount << ": Visualization running" << std::endl;
            // }
            
            // Limit frame rate
            std::this_thread::sleep_for(std::chrono::milliseconds(16)); // ~60 FPS
        }
        
        std::cout << "Visualization test complete" << std::endl;
        
        // Clean up
        visManager.shutdown();
        simManager.shutdown();
        
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