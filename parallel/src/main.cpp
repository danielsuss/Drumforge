#include "cuda_memory_manager.h"
#include <iostream>

int main(int argc, char* argv[]) {
    std::cout << "DrumForge Parallel - Starting up..." << std::endl;
    
    try {
        // Initialize the CUDA memory manager
        drumforge::CudaMemoryManager& memoryManager = drumforge::CudaMemoryManager::getInstance();
        memoryManager.initialize();
        
        std::cout << "CUDA Memory Manager initialized successfully!" << std::endl;
        
        // Clean up
        memoryManager.shutdown();
        
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