#include "ShaderManager.h"

Shader* ShaderManager::load(const std::string& name, const char* vertex_path, const char* fragment_path) {
	auto shader = std::make_unique<Shader>(vertex_path, fragment_path);
	Shader* raw = shader.get();
	_shaders[name] = std::move(shader);
	return raw;
}

Shader* ShaderManager::get(const std::string& name) const {
	auto it = _shaders.find(name);
	if (it == _shaders.end()) return nullptr;
	return it->second.get();
}

Shader* ShaderManager::get_or_load(const std::string& name, const char* vertex_path, const char* fragment_path) {
	if (Shader* existing = get(name)) return existing;
	return load(name, vertex_path, fragment_path);
}
