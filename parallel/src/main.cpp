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
        if (!visManager.initialize(1280, 720, "DrumForge - Visualization Test")) {
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
        
        // Test the visualization loop
        std::cout << "Starting visualization test loop..." << std::endl;
        
        // Fixed timestep for simulation
        const float timestep = 1.0f / 60.0f;  // ~60 FPS
        
        // Connect the membrane to the input handler for click interaction
        auto inputHandler = visManager.getInputHandler();
        if (inputHandler) {
            inputHandler->connectMembrane(membrane);
            std::cout << "Membrane connected to input handler - click on the membrane to apply impulses" << std::endl;
        }
        
        // Main loop
        while (!visManager.shouldClose()) {
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
            CHECK_CUDA_ERRORS();
            
            // Render frame
            visManager.beginFrame();
            visManager.renderComponents(simManager);
            visManager.endFrame();
            CHECK_CUDA_ERRORS();
            
            // Limit frame rate 
            std::this_thread::sleep_for(std::chrono::milliseconds(16)); // ~60 FPS
        }
        
        std::cout << "Visualization test complete" << std::endl;
        
        // Clean up - order is important for clean shutdown
        // First, clean up visualization resources
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