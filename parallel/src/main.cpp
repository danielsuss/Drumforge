#include "cuda_memory_manager.h"
#include "simulation_manager.h"
#include <iostream>

int main(int argc, char* argv[]) {
    std::cout << "DrumForge Parallel - Starting up..." << std::endl;
    
    try {
        // Get the simulation manager
        drumforge::SimulationManager& simManager = drumforge::SimulationManager::getInstance();
        
        // Initialize the simulation
        simManager.initialize();
        
        std::cout << "SimulationManager initialized successfully!" << std::endl;
        
        // Modify some simulation parameters
        drumforge::SimulationParameters params = simManager.getParameters();
        params.timeScale = 2.0f;
        simManager.updateParameters(params);
        
        std::cout << "Simulation parameters updated (timeScale = 2.0)" << std::endl;
        
        // Run a few steps of an empty simulation (will use default timestep)
        for (int i = 0; i < 5; i++) {
            simManager.advance(0.016f); // ~60 FPS
            std::cout << "Simulation advanced, step " << (i + 1) << std::endl;
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