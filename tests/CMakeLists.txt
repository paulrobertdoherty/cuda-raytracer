add_executable(ray_math_test ray_math.cu ../src/cuda_errors.cpp)
target_link_libraries(ray_math_test PRIVATE glm::glm CUDA::cudart)
set_target_properties(ray_math_test PROPERTIES CUDA_STANDARD 17 CXX_STANDARD 17)
if(MSVC)
	target_compile_options(ray_math_test PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/Zc:preprocessor>")
endif()
add_test(NAME ray_math COMMAND ray_math_test)

if(HEADLESS)
	find_package(Python3 REQUIRED COMPONENTS Interpreter)
	add_test(NAME render_regression
		COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/render_regression.py
			$<TARGET_FILE:${PROJECT_NAME}>)
endif()
