#include "ShaderManager.h"

#include <filesystem>

#ifdef _WIN32
#include <windows.h>
#else
#include <unistd.h>
#include <limits.h>
#endif

// Resolve a path relative to the executable's directory instead of the
// current working directory, so shader files found by CMake's file(COPY ...)
// load correctly regardless of where the user launched the binary from.
static std::string resolve_from_exe(const char* rel) {
	std::filesystem::path input(rel);
	if (input.is_absolute()) return std::string(rel);

	std::filesystem::path exe_dir;
#ifdef _WIN32
	char buf[MAX_PATH];
	DWORD n = GetModuleFileNameA(nullptr, buf, MAX_PATH);
	if (n == 0 || n >= MAX_PATH) return std::string(rel);
	exe_dir = std::filesystem::path(buf).parent_path();
#else
	char buf[PATH_MAX];
	ssize_t n = readlink("/proc/self/exe", buf, sizeof(buf) - 1);
	if (n <= 0) return std::string(rel);
	buf[n] = '\0';
	exe_dir = std::filesystem::path(buf).parent_path();
#endif
	return (exe_dir / input.relative_path()).lexically_normal().string();
}

Shader* ShaderManager::load(const std::string& name, const char* vertex_path, const char* fragment_path) {
	std::string v = resolve_from_exe(vertex_path);
	std::string f = resolve_from_exe(fragment_path);
	auto shader = std::make_unique<Shader>(v.c_str(), f.c_str());
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
