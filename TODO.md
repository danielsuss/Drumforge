# DrumForge Step 4: Membrane Visualization Tasks

## Core Implementation Tasks
- [ ] Make `isInsideCircle()` method public in `membrane.h`
- [ ] Modify `main.cpp` to include grid visualization
- [ ] Create vertex buffer for membrane points
- [ ] Setup simple orthographic projection
- [ ] Implement point rendering with different colors for active/inactive regions

## File Creation & Updates
- [x] Create `shaders/point_vertex.glsl`
- [x] Create `shaders/point_fragment.glsl`
- [x] Create `include/shader.h`
- [x] Create `src/shader.cpp`
- [ ] Update `CMakeLists.txt` to include new shader files

## Rendering Pipeline
- [ ] Setup VAO/VBO for grid points
- [ ] Add vertex attribute configuration
- [ ] Implement shader activation in render loop
- [ ] Add point drawing logic based on membrane boundary
- [ ] Configure point size and colors

## Testing & Refinement
- [ ] Verify that circular membrane boundary appears correctly
- [ ] Adjust point size for better visibility if needed
- [ ] Check performance with different grid resolutions
- [ ] Add window resize handling

## Optional Enhancements
- [ ] Add color variation based on grid position
- [ ] Implement simple camera controls (zoom/pan)
- [ ] Add wireframe visualization mode option