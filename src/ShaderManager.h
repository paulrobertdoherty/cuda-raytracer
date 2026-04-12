#pragma once

#include "Shader.h"

#include <memory>
#include <string>
#include <unordered_map>

// Named collection of compiled GL shader programs. Callers ask for shaders by
// name and the manager compiles them lazily on first request (so unused
// shaders don't get built).
class ShaderManager {
public:
	ShaderManager() = default;

	// Non-copyable (holds unique_ptrs) but movable.
	ShaderManager(const ShaderManager&) = delete;
	ShaderManager& operator=(const ShaderManager&) = delete;
	ShaderManager(ShaderManager&&) = default;
	ShaderManager& operator=(ShaderManager&&) = default;

	// Compile and register a shader under `name`. Overwrites any existing
	// entry with the same name.
	Shader* load(const std::string& name, const char* vertex_path, const char* fragment_path);

	// Return the shader registered under `name`, or nullptr if none.
	Shader* get(const std::string& name) const;

	// Convenience: fetch if loaded, otherwise compile with the given paths.
	Shader* get_or_load(const std::string& name, const char* vertex_path, const char* fragment_path);

private:
	std::unordered_map<std::string, std::unique_ptr<Shader>> _shaders;
};
