cmake_minimum_required (VERSION 3.24)

# Enable Hot Reload for MSVC compilers if supported.
if (POLICY CMP0141)
  cmake_policy(SET CMP0141 NEW)
  set(CMAKE_MSVC_DEBUG_INFORMATION_FORMAT "$<IF:$<AND:$<C_COMPILER_ID:MSVC>,$<CXX_COMPILER_ID:MSVC>>,$<$<CONFIG:Debug,RelWithDebInfo>:EditAndContinue>,$<$<CONFIG:Debug,RelWithDebInfo>:ProgramDatabase>>")
endif()

project ("cuda-raytracer" LANGUAGES CUDA CXX C)

# Copy the shader source files to the build directory
file(COPY ${CMAKE_SOURCE_DIR}/src/shaders DESTINATION .)

# The configurations listed below were modified and taken from https://github.com/Shot511/OpenGLSampleCmake

# Add source files
file(GLOB_RECURSE SOURCE_FILES 
	${CMAKE_SOURCE_DIR}/src/*.c
	${CMAKE_SOURCE_DIR}/src/*.cpp
	${CMAKE_SOURCE_DIR}/src/raytracer/*.cu)
	
# Add header files
file(GLOB_RECURSE HEADER_FILES 
	${CMAKE_SOURCE_DIR}/src/*.h
	${CMAKE_SOURCE_DIR}/src/*.hpp
	${CMAKE_SOURCE_DIR}/src/raytracer/*.h)

# Add source to this project's executable.
add_executable(${PROJECT_NAME} ${HEADER_FILES} ${SOURCE_FILES})

# Set CUDA architecture to the host's GPU architecture (default: native, override with -DCMAKE_CUDA_ARCHITECTURES=XX)
if(DEFINED CMAKE_CUDA_ARCHITECTURES)
    set_target_properties(${PROJECT_NAME} PROPERTIES CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}")
else()
    set_target_properties(${PROJECT_NAME} PROPERTIES CUDA_ARCHITECTURES native)
endif()

# CUDA 13+ CCCL requires C++17
set_target_properties(${PROJECT_NAME} PROPERTIES
	CXX_STANDARD 17
	CUDA_STANDARD 17)

# Add .lib files
link_directories(${CMAKE_SOURCE_DIR}/lib)

# We need a CMAKE_DIR with some code to find external dependencies
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/")

#######################################
# LOOK for the packages that we need! #
#######################################

# OpenGL
set(OpenGL_GL_PREFERENCE GLVND)
find_package(OpenGL REQUIRED)

# GLM
find_package(GLM REQUIRED)
message(STATUS "GLM included at ${GLM_INCLUDE_DIR}")

# GLFW
find_package(GLFW3 REQUIRED)
message(STATUS "Found GLFW3 in ${GLFW3_INCLUDE_DIR}")

# GLAD
add_library(GLAD "thirdparty/glad.c")

# Dear ImGui + ImGuiFileDialog
add_library(IMGUI STATIC
	thirdparty/imgui/imgui.cpp
	thirdparty/imgui/imgui_draw.cpp
	thirdparty/imgui/imgui_tables.cpp
	thirdparty/imgui/imgui_widgets.cpp
	thirdparty/imgui/backends/imgui_impl_glfw.cpp
	thirdparty/imgui/backends/imgui_impl_opengl3.cpp
	thirdparty/ImGuiFileDialog/ImGuiFileDialog.cpp
)
target_include_directories(IMGUI PUBLIC
	"${CMAKE_SOURCE_DIR}/thirdparty/imgui"
	"${CMAKE_SOURCE_DIR}/thirdparty/imgui/backends"
	"${CMAKE_SOURCE_DIR}/thirdparty/ImGuiFileDialog"
	"${CMAKE_SOURCE_DIR}/include"
	"${CMAKE_SOURCE_DIR}/include/glad"
)
target_link_libraries(IMGUI ${GLFW3_LIBRARY} ${OPENGL_LIBRARY})

# CUDA
find_package(CUDAToolkit REQUIRED)
include_directories("${CUDAToolkit_INCLUDE_DIRS}")
# CUDA 12+ moved Thrust/CUB/libcudacxx into a CCCL subdirectory
if(EXISTS "${CUDAToolkit_INCLUDE_DIRS}/cccl")
	include_directories("${CUDAToolkit_INCLUDE_DIRS}/cccl")
endif()

# Put all libraries into a variable
set(LIBS ${GLFW3_LIBRARY} ${OPENGL_LIBRARY} GLAD IMGUI ${CMAKE_DL_LIBS} CUDA::cudart CUDA::curand)

# Define the include DIRs
include_directories(
	"${CMAKE_SOURCE_DIR}/src"
	"${CMAKE_SOURCE_DIR}/include"
	"${CMAKE_SOURCE_DIR}/include/glad"
	"${CMAKE_SOURCE_DIR}/include/glm"
	"${CMAKE_SOURCE_DIR}/thirdparty"
	"${CMAKE_SOURCE_DIR}/thirdparty/imgui"
	"${CMAKE_SOURCE_DIR}/thirdparty/imgui/backends"
	"${CMAKE_SOURCE_DIR}/thirdparty/ImGuiFileDialog"
)

# Prevent Windows min/max/small macros from conflicting with C++ standard library
if(WIN32)
	target_compile_definitions(${PROJECT_NAME} PRIVATE NOMINMAX WIN32_LEAN_AND_MEAN)
endif()

# Define the link libraries
target_link_libraries(${PROJECT_NAME} ${LIBS})

# Create virtual folders to make it look nicer in VS
if(MSVC_IDE)
	# Macro to preserve source files hierarchy in the IDE
	macro(GroupSources curdir)
		file(GLOB children RELATIVE ${PROJECT_SOURCE_DIR}/${curdir} ${PROJECT_SOURCE_DIR}/${curdir}/*)

		foreach(child ${children})
			if(IS_DIRECTORY ${PROJECT_SOURCE_DIR}/${curdir}/${child})
				GroupSources(${curdir}/${child})
			else()
				string(REPLACE "/" "\\" groupname ${curdir})
				string(REPLACE "src" "Sources" groupname ${groupname})
				source_group(${groupname} FILES ${PROJECT_SOURCE_DIR}/${curdir}/${child})
			endif()
		endforeach()
	endmacro()

	# Run macro
	GroupSources(src)
endif()


function(CUDA_CONVERT_FLAGS EXISTING_TARGET)
	get_property(old_flags TARGET ${EXISTING_TARGET} PROPERTY INTERFACE_COMPILE_OPTIONS)
	if(NOT "${old_flags}" STREQUAL "")
		string(REPLACE ";" "," CUDA_flags "${old_flags}")
		set_property(TARGET ${EXISTING_TARGET} PROPERTY INTERFACE_COMPILE_OPTIONS
			"$<$<BUILD_INTERFACE:$<COMPILE_LANGUAGE:CXX>>:${old_flags}>$<$<BUILD_INTERFACE:$<COMPILE_LANGUAGE:CUDA>>:-Xcompiler=${CUDA_flags}>"
			)
	endif()
endfunction()

CUDA_CONVERT_FLAGS(${PROJECT_NAME})

# CUDA 13+ CCCL requires the standard conforming preprocessor
if(MSVC)
	target_compile_options(${PROJECT_NAME} PRIVATE
		"$<$<COMPILE_LANGUAGE:CXX>:/Zc:preprocessor>"
		"$<$<COMPILE_LANGUAGE:C>:/Zc:preprocessor>"
		"$<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/Zc:preprocessor>")
endif()

# check for cache variable set in cmakepresets called IS_CUDA_DEBUG
if(IS_CUDA_DEBUG) 
	target_compile_options(${PROJECT_NAME} PRIVATE "$<$<AND:$<CONFIG:Debug>,$<COMPILE_LANGUAGE:CUDA>>:-G>")
	message(STATUS "CUDA_DEBUG is ON")
endif()