# Drumforge

Drumforge is a physically-based drum synthesis system powered by CUDA GPU acceleration. It simulates drum physics in real-time using the Finite Difference Time Domain (FDTD) method for the membrane and modal synthesis for the shell resonance.

[![Drumforge](https://img.youtube.com/vi/7DiiWhnudG8/maxresdefault.jpg)](https://youtu.be/7DiiWhnudG8?si=LHFoFV_swRQLIIEZ)

<div align="center">
  <em>Click to watch the demo on YouTube</em>
</div>

## Features

- GPU-accelerated membrane simulation
- Real-time visualization of membrane vibrations
- Interactive strike input via mouse clicks
- Adjustable physical parameters (tension, damping, etc.)
- Configurable virtual microphones
- Audio recording with WAV export
- Multiple shell material presets

## Dependencies

- CUDA Toolkit 12.8 (or compatible version)
- OpenGL
- GLFW3
- GLEW
- GLM
- CMake 3.31.6 or higher

## Building from Source

### 1. Clone the repository with submodules

```bash
git clone https://github.com/danielsuss/Drumforge.git
cd Drumforge
git submodule update --init --recursive
```

### 2. Install dependencies

#### Ubuntu/Debian:

```bash
sudo apt update
sudo apt install build-essential cmake libglfw3-dev libglew-dev libglm-dev
```

#### Arch Linux:

```bash
sudo pacman -S base-devel cmake glfw-x11 glew glm
```

#### macOS (using Homebrew):

```bash
brew install cmake glfw glew glm
```

#### Windows:

Install the dependencies using vcpkg or download the binaries for each library.

### 3. Install CUDA Toolkit

Download and install the CUDA Toolkit 12.8 from the [NVIDIA website](https://developer.nvidia.com/cuda-downloads).

### 4. Update CUDA paths in CMakeLists.txt

If your CUDA installation is not in the default location (`/usr/local/cuda-12.8`), you'll need to update the paths in `CMakeLists.txt`. Modify the following lines:

```cmake
set(CMAKE_CUDA_COMPILER /usr/local/cuda-12.8/bin/nvcc)
```

And in the include directories:

```cmake
target_include_directories(cuda_test PRIVATE
    /usr/local/cuda-12.8/include
)

target_include_directories(drumforge_parallel PRIVATE
    ${CMAKE_SOURCE_DIR}/parallel/include
    ${CMAKE_SOURCE_DIR}/vendor/imgui
    ${CMAKE_SOURCE_DIR}/vendor/imgui/backends
    /usr/local/cuda-12.8/include
)
```

### 5. Configure and build

```bash
mkdir build
cd build
cmake ..
make
```

## Running Drumforge

### Running the parallel version:

```bash
./drumforge_parallel
```

To enable CUDA-OpenGL interoperability (if supported by your system):

```bash
./drumforge_parallel --enable-interop
```

## Usage Instructions

1. Start the application with the command above
2. Configure initial settings in the setup screen:
   - Grid size
   - Membrane radius, tension, and damping
   - Shell material and dimensions
3. Click "Start Simulation" to begin
4. Use WASD/QE keys to navigate the camera
5. Click on the membrane to apply strikes
6. Use the control panel to adjust parameters, add/remove microphones, or record audio

### Keyboard Controls

- **W/S**: Move camera forward/backward
- **A/D**: Move camera left/right
- **Q/E**: Move camera down/up
- **Arrow Keys**: Pan camera direction
- **R**: Reset camera position
- **H**: Show help
- **ESC**: Exit application

## Troubleshooting

If you encounter build issues:

1. Ensure your CUDA installation matches the paths in CMakeLists.txt
2. Check that your GPU supports CUDA and has compatible compute capability
3. Make sure all dependencies are properly installed

For detailed logging, run in a terminal and check the console output.
