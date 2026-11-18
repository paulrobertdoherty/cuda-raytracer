add_executable(ray_math_test ray_math.cu ../src/cuda_errors.cpp)
target_link_libraries(ray_math_test PRIVATE glm::glm CUDA::cudart)
set_target_properties(ray_math_test PROPERTIES CUDA_STANDARD 17 CXX_STANDARD 17)
if(MSVC)
	target_compile_options(ray_math_test PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/Zc:preprocessor>")
endif()
add_test(NAME ray_math COMMAND ray_math_test)

add_executable(render_state_test render_state.cpp)
target_compile_features(render_state_test PRIVATE cxx_std_17)
add_test(NAME render_state COMMAND render_state_test)

if(HEADLESS)
	add_executable(scene_updates_test scene_updates.cpp)
	target_link_libraries(scene_updates_test PRIVATE headless_renderer)
	add_test(NAME scene_updates COMMAND scene_updates_test "${CMAKE_CURRENT_SOURCE_DIR}/data")

	find_package(Python3 REQUIRED COMPONENTS Interpreter)
	add_test(NAME render_regression
		COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/render_regression.py
			$<TARGET_FILE:${PROJECT_NAME}>)
endif()
