#include "ObjLoader.h"

#define TINYOBJLOADER_IMPLEMENTATION
#include "tiny_obj_loader.h"

#include <glm/glm.hpp>

#include <iostream>
#include <unordered_map>
#include <algorithm>

namespace ObjLoader {

namespace {

struct VertexKey {
	int pos;
	int normal;
	int uv;
	bool operator==(const VertexKey& other) const {
		return pos == other.pos && normal == other.normal && uv == other.uv;
	}
};

struct VertexKeyHash {
	size_t operator()(const VertexKey& k) const {
		size_t h = std::hash<int>()(k.pos);
		h ^= std::hash<int>()(k.normal) + 0x9e3779b9 + (h << 6) + (h >> 2);
		h ^= std::hash<int>()(k.uv) + 0x9e3779b9 + (h << 6) + (h >> 2);
		return h;
	}
};

} // namespace

std::unique_ptr<Mesh> load(const std::string& path) {
	tinyobj::attrib_t attrib;
	std::vector<tinyobj::shape_t> shapes;
	std::vector<tinyobj::material_t> materials;
	std::string warn;
	std::string err;

	// Extract the directory from the OBJ path so tinyobj can find the .mtl file
	std::string mtl_basedir;
	auto last_slash = path.find_last_of("/\\");
	if (last_slash != std::string::npos) {
		mtl_basedir = path.substr(0, last_slash + 1);
	}

	// triangulate=true, default_vcols_fallback=false
	bool ok = tinyobj::LoadObj(&attrib, &shapes, &materials, &warn, &err,
	                            path.c_str(),
	                            mtl_basedir.empty() ? nullptr : mtl_basedir.c_str(),
	                            true);
	if (!warn.empty()) {
		std::cerr << "[ObjLoader] " << warn << std::endl;
	}
	if (!err.empty()) {
		std::cerr << "[ObjLoader] " << err << std::endl;
	}
	if (!ok) {
		return nullptr;
	}

	auto mesh = std::make_unique<Mesh>();

	std::unordered_map<VertexKey, unsigned int, VertexKeyHash> index_cache;

	for (const auto& shape : shapes) {
		const auto& mesh_data = shape.mesh;
		size_t index_offset = 0;
		for (size_t f = 0; f < mesh_data.num_face_vertices.size(); f++) {
			int fv = mesh_data.num_face_vertices[f];
			// Should always be 3 because we requested triangulation, but be safe.
			if (fv != 3) { index_offset += fv; continue; }

			// Precompute flat normal as a fallback.
			glm::vec3 face_pos[3];
			for (int v = 0; v < 3; v++) {
				auto idx = mesh_data.indices[index_offset + v];
				face_pos[v] = glm::vec3(
					attrib.vertices[3 * idx.vertex_index + 0],
					attrib.vertices[3 * idx.vertex_index + 1],
					attrib.vertices[3 * idx.vertex_index + 2]);
			}
			glm::vec3 flat_normal = glm::normalize(
				glm::cross(face_pos[1] - face_pos[0], face_pos[2] - face_pos[0]));

			for (int v = 0; v < 3; v++) {
				auto idx = mesh_data.indices[index_offset + v];
				VertexKey key{idx.vertex_index, idx.normal_index, idx.texcoord_index};
				auto it = index_cache.find(key);
				if (it != index_cache.end()) {
					mesh->indices.push_back(it->second);
					continue;
				}

				MeshVertex out;
				out.position = face_pos[v];

				if (idx.normal_index >= 0) {
					out.normal = glm::vec3(
						attrib.normals[3 * idx.normal_index + 0],
						attrib.normals[3 * idx.normal_index + 1],
						attrib.normals[3 * idx.normal_index + 2]);
				} else {
					out.normal = flat_normal;
				}

				if (idx.texcoord_index >= 0) {
					out.uv = glm::vec2(
						attrib.texcoords[2 * idx.texcoord_index + 0],
						attrib.texcoords[2 * idx.texcoord_index + 1]);
				} else {
					out.uv = glm::vec2(0.0f);
				}

				unsigned int new_idx = (unsigned int)mesh->vertices.size();
				mesh->vertices.push_back(out);
				mesh->indices.push_back(new_idx);
				index_cache[key] = new_idx;
			}

			index_offset += 3;
		}
	}

	std::cout << "[ObjLoader] Loaded " << path << " — "
	          << mesh->vertices.size() << " vertices, "
	          << mesh->indices.size() / 3 << " triangles" << std::endl;
	return mesh;
}

} // namespace ObjLoader
