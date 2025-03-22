#include <iostream>

// External function declaration
extern "C" bool testCuda();

int main() {
    std::cout << "Running CUDA test..." << std::endl;
    
    if (testCuda()) {
        std::cout << "✅ CUDA test passed!" << std::endl;
        return 0;
    } else {
        std::cout << "❌ CUDA test failed!" << std::endl;
        return 1;
    }
}