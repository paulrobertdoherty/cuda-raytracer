#pragma once

#include "Mesh.h"

#include <memory>
#include <string>

namespace ObjLoader {

// Load a .obj file, flatten all shapes into one indexed mesh, and return it.
// Generates flat per-face normals if the .obj lacks them. Returns nullptr on
// failure.
std::unique_ptr<Mesh> load(const std::string& path);

} // namespace ObjLoader
