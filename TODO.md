# TODO
- [x] Convert raw device pointer usage to use `thrust::device_ptr`
- [x] Make use of `thrust::device_vector` instead of raw arrays for the world list
    - This does not seem to be possible since `thrust::device_vector` is designed to manage device memory **from the host**
- [x] Fix loss of data warnings in Camera class
- [x] Fix exception thrown when resizing window
- [x] Fix CUDA memory leak related to `cudaGraphicsGLRegisterBuffer` in `Quad.cpp:75`
- [x] Check out how the OpenGL and CUDA interop works since right now a new frame buffer is allocated each frame (seems pretty bad for performance)
- [x] Enable BVH acceleration (integrate existing `BVHNode` into `World::hit()`)
- [ ] Add a GUI to display information (imgui)

## Potential Improvements

### Rendering
- [x] Make samples-per-pixel configurable instead of hardcoded to 3 in `kernel.cu`
- [x] Adapt SPP dynamically based on frame time budget
- [x] Add emissive material for area lights (currently only sky gradient illumination)
- [x] Add temporal reprojection so accumulation doesn't fully reset on camera movement
- [x] Add rectangle and triangle primitives (needed for area lights, boxes, meshes)

### Performance
- [x] Replace single-thread `set_device_camera` kernel with `cudaMemcpyAsync`
- [x] Use `CMAKE_CUDA_ARCHITECTURES = native` instead of hardcoded `sm_86` for portability

### Code Quality
- [x] Centralize magic constants (window size, max depth, SPP, FOV — now CLI parameters)
- [ ] Make remaining magic constants configurable (world capacity 20, accumulation blend rate 500)
- [ ] Make `World` capacity dynamic instead of fixed at 20 objects
