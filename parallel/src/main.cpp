#include "cuda_memory_manager.h"
#include "simulation_manager.h"
#include "membrane_component.h"
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
        // Get the simulation manager
        drumforge::SimulationManager& simManager = drumforge::SimulationManager::getInstance();
        
        // Initialize the simulation
        simManager.initialize();
        CHECK_CUDA_ERRORS();
        
        std::cout << "SimulationManager initialized successfully!" << std::endl;
        
        // Modify simulation parameters - use smaller grid for testing
        drumforge::SimulationParameters params = simManager.getParameters();
        params.timeScale = 1.0f;
        params.gridSizeX = 64;  // Reduced from 128
        params.gridSizeY = 64;  // Reduced from 128
        params.gridSizeZ = 16;  // Reduced from 32
        params.cellSize = 0.1f; // Increased from 0.05f - fewer cells to manage
        simManager.updateParameters(params);
        
        std::cout << "Simulation parameters updated (gridSize = " 
                  << params.gridSizeX << "x" << params.gridSizeY << "x" << params.gridSizeZ
                  << ", cellSize = " << params.cellSize << ")" << std::endl;
        
        // Create a membrane component with more modest parameters for testing
        std::cout << "Creating membrane component..." << std::endl;
        auto membrane = std::make_shared<drumforge::MembraneComponent>(
            "Drumhead", 
            2.0f,    // Reduced radius from 5.0 to 2.0 units
            10.0f,   // Reduced tension from 50.0 to 10.0
            0.01f    // Slightly increased damping
        );
        
        std::cout << "Membrane component created" << std::endl;
        
        // Make sure CudaMemoryManager is initialized before adding component
        drumforge::CudaMemoryManager::getInstance().initialize();
        CHECK_CUDA_ERRORS();
        
        std::cout << "About to add membrane to simulation..." << std::endl;
        // Add the membrane to the simulation
        simManager.addComponent(membrane);
        CHECK_CUDA_ERRORS();
        
        std::cout << "Membrane component added to simulation" << std::endl;
        
        std::cout << "About to run membrane.initialize() manually..." << std::endl;
        // Manually initialize the membrane component
        membrane->initialize();
        CHECK_CUDA_ERRORS();
        
        std::cout << "Membrane initialization complete" << std::endl;
        
        // Try basic operations first
        try {
            // Get membrane dimensions
            int width = membrane->getMembraneWidth();
            int height = membrane->getMembraneHeight();
            float radius = membrane->getRadius();
            
            std::cout << "Membrane dimensions: " << width << "x" << height 
                      << " (radius: " << radius << ")" << std::endl;
            
            // Retrieve heights array (tests CUDA to CPU transfer)
            std::cout << "Getting membrane heights..." << std::endl;
            const auto& heights = membrane->getHeights();
            std::cout << "Retrieved " << heights.size() << " height values" << std::endl;
            
            // Check central height (should be 0 initially)
            int centralX = width / 2;
            int centralY = height / 2;
            float centralHeight = membrane->getHeight(centralX, centralY);
            std::cout << "Initial central height: " << centralHeight << std::endl;
            
            // Try resetting the membrane (should be a no-op since it's already flat)
            std::cout << "Resetting membrane..." << std::endl;
            membrane->reset();
            CHECK_CUDA_ERRORS();
            std::cout << "Membrane reset complete" << std::endl;
            
            // Now try applying an impulse
            std::cout << "Applying impulse..." << std::endl;
            membrane->applyImpulse(0.5f, 0.5f, 0.05f); // Lower strength
            CHECK_CUDA_ERRORS();
            std::cout << "Applied impulse to membrane center" << std::endl;
            
            // Run a single simulation step
            const float timestep = 1.0f / 60.0f;  // ~60 FPS
            std::cout << "Running single simulation step..." << std::endl;
            simManager.advance(timestep);
            CHECK_CUDA_ERRORS();
            std::cout << "Simulation step complete" << std::endl;
            
            // Check if the height changed
            centralHeight = membrane->getHeight(centralX, centralY);
            std::cout << "Central height after impulse and one step: " << centralHeight << std::endl;
            
            // If we get here without segfault, try a few more steps
            // Run several steps of the simulation
            const int numSteps = 10; // Reduced from 60
            
            std::cout << "Starting simulation for " << numSteps << " steps..." << std::endl;
            
            // Print header for height values
            std::cout << "Step\tCentral Height" << std::endl;
            std::cout << "-------------------------" << std::endl;
            
            for (int i = 0; i < numSteps; i++) {
                // Advance the simulation
                simManager.advance(timestep);
                CHECK_CUDA_ERRORS();
                
                // Get and print the height at the center of the membrane
                centralHeight = membrane->getHeight(centralX, centralY);
                std::cout << i + 1 << "\t" << centralHeight << std::endl;
            }
            
            std::cout << "Simulation complete" << std::endl;
        }
        catch (const std::exception& e) {
            std::cerr << "Error during membrane operations: " << e.what() << std::endl;
        }
        
        // Clean up
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