# DrumForge: GPU-Accelerated Drum Simulation

A physically-based drum synthesizer that leverages CUDA for high-performance simulation of drum acoustics.

## Project Overview

DrumForge aims to create physically accurate drum sound synthesis by balancing computational efficiency with sound quality through:

1. Highly parallel GPU-accelerated physics for accurate acoustic simulation
2. Modular component architecture for complex drum modeling
3. Real-time visualization and sound generation

## Architecture

DrumForge employs a modular component-based architecture that allows for scalable simulation of complex drum systems:

```
┌───────────────────────────────────────┐
│             SimulationManager         │
├───────────────────────────────────────┤
│ - Coordinates component interactions  │
│ - Manages simulation time stepping    │
│ - Handles global parameters           │
└─────────────┬─────────────────────────┘
              │
              ▼
┌───────────────────────────────────────┐
│        CudaMemoryManager              │
├───────────────────────────────────────┤
│ - Allocates/frees GPU memory          │
│ - Manages OpenGL interoperability     │
│ - Handles device synchronization      │
└─────────────┬─────────────────────────┘
              │
              ▼
┌─────────────┴─────────────────────────┐
│         ComponentInterface            │
├───────────────────────────────────────┤
│ - Common interface for all components │
│ - Defines init/update/render methods  │
│ - Specifies coupling mechanisms       │
└───┬───────────────┬───────────────┬───┘
    │               │               │
    ▼               ▼               ▼
┌─────────┐    ┌─────────┐    ┌─────────┐
│ Membrane │    │AirCavity│    │DrumShell│
│Component │    │Component│    │Component│
└─────────┘    └─────────┘    └─────────┘
```

### Core Components

- **SimulationManager**: Coordinates the simulation of all components and handles their interactions
- **CudaMemoryManager**: Provides unified memory management for all GPU buffers
- **ComponentInterface**: A standard interface that all drum components implement
- **Membrane Component**: Simulates the vibrating membranes using FDTD methods
- **AirCavity Component**: Models the acoustic behavior of air inside the drum
- **DrumShell Component**: Simulates the resonant properties of the drum shell

### Technical Approach

- **CUDA-OpenGL Interoperability**: Direct GPU-to-visualization pipeline without CPU transfers
- **Finite Difference Time Domain (FDTD)**: For accurate wave propagation simulation
- **Component Coupling**: Interfaces for exchanging forces and boundary conditions between components

## Current Features

- CUDA-accelerated membrane simulation
- OpenGL visualization of membrane dynamics
- User interaction via mouse and keyboard
- Real-time parameter adjustment
- Direct mallet strike simulation

## Planned Features

- Multiple coupled membranes (top and bottom)
- Internal air cavity simulation
- Acoustic radiation modeling
- Import of 3D models from Blender
- Advanced material properties
- Programmable strike patterns
- Audio export functionality

## Tech Stack

- **Core**: C++, CUDA, OpenGL
- **Libraries**:
  - GLFW (window management)
  - Dear ImGui (user interface)
  - GLM (math operations)
  - RtAudio (planned for audio output)
  - libsndfile (planned for audio export)
- **Build System**: CMake with CUDA support

## Building and Running

### Prerequisites

- CMake 3.31.6 or higher
- CUDA Toolkit 12.8 or higher
- OpenGL development libraries
- GLFW3 development libraries
- GLEW development libraries
- GLM development libraries
- C++ compiler with C++23 support

### Build Instructions

```bash
# Clone the repository
git clone https://github.com/yourusername/drumforge.git
cd drumforge

# Create build directory
mkdir build
cd build

# Configure and build
cmake ..
cmake --build .

# Run the application
./drumforge

# Run the CUDA tests
./cuda_test
```

## Controls

- **Mouse Left-Click**: Strike the drum membrane
- **WASD, QE**: Move camera
- **Arrow Keys**: Pan camera view
- **1, 2, 3**: Toggle visibility of components
- **R**: Reset membrane to flat state
- **H**: Show help message
- **ESC**: Exit application

## References

- Bilbao, S., & Webb, C. (2012). Timpani Drum Synthesis in 3D on GPGPUs
- NESS (Next Generation Sound Synthesis) project at the University of Edinburgh
