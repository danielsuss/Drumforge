# GPU-Accelerated DrumForge Prototype

This repository contains a prototype implementation for simulating drum membranes and resonance using GPU acceleration.

## Project Overview

The goal is to create a physically accurate drum sound synthesis system that balances computational efficiency with sound quality by:

1. Using GPU-accelerated physics for the membrane vibration (high-fidelity simulation)
2. Approximating the drum shell/body response (efficient resonator model)

## Technical Approach

### Core Simulation Method
- **Membrane Physics**: Finite Difference Time Domain (FDTD) method implemented on CUDA
- **Body Resonance**: Modal synthesis approach with simplified coupling
- **Sound Generation**: Direct conversion of membrane displacement to audio signal

### Tech Stack
- **Core**: C++, CUDA, OpenGL
- **Libraries**:
  - GLFW (window management)
  - Dear ImGui (user interface)
  - RtAudio (real-time audio playback)
  - libsndfile (audio file export and analysis)
  - GLM (math operations)
- **Build System**: CMake with Git submodules for dependencies

## Prototype Scope

For the initial prototype, we're focusing on:

- Procedurally generated circular membrane (no Blender import)
- Basic visualization of membrane deformation
- Simple user interface for parameter adjustment
- Real-time audio output
- Simple mallet strike simulation

## Future Extensions

After the prototype is functional:
- 3D model import from Blender
- More sophisticated material models
- Advanced excitation methods
- Improved coupling between membrane and body
- Support for different drum types

## Development Plan

1. Set up project structure and build system
2. Implement basic CUDA kernel for membrane simulation
3. Add OpenGL visualization
4. Create user interface with Dear ImGui
5. Integrate audio output
6. Refine and optimize the simulation
7. Add parameter controls and presets

## Building and Running

### Prerequisites
- CMake 3.15 or higher
- CUDA Toolkit
- OpenGL development libraries
- GLFW3 development libraries
- RtAudio development libraries
- libsndfile development libraries
- C++ compiler with C++17 support

### Build Instructions
```bash
# Clone the repository
git clone https://github.com/yourusername/drumforge.git
cd drumforge

# Initialize and update submodules
git submodule init
git submodule update

# Create build directory
mkdir build
cd build

# Configure and build
cmake ..
cmake --build .

# Run the tests
./drumforge_test
```

### Test Output
The test program will verify that all required components are functioning correctly:
- CUDA functionality (GPU device detection and basic computation)
- OpenGL/GLFW window creation
- ImGui interface rendering
- RtAudio sound playback (you should hear a short 440Hz tone)
- libsndfile file export (creates a test_sine.wav file)

A successful test will display "✅ All tests completed successfully!" in the console.

## References

- Bilbao, S., & Webb, C. (2012). Timpani Drum Synthesis in 3D on GPGPUs
- NESS (Next Generation Sound Synthesis) project at the University of Edinburgh