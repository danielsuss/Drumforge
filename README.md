# GPU-Accelerated Drum Simulation Prototype

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
  - miniaudio (audio output and file export)
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

## References

- Bilbao, S., & Webb, C. (2012). Timpani Drum Synthesis in 3D on GPGPUs
- NESS (Next Generation Sound Synthesis) project at the University of Edinburgh