#include "Mesh.h"
#include "AssetRevision.h"

void Mesh::upload() {
	_revision = next_asset_revision();
	if (vertices.empty() || indices.empty()) {
		_index_count = 0;
		return;
	}

	// Compute bounding box.
	local_min = vertices[0].position;
	local_max = vertices[0].position;
	for (const auto& v : vertices) {
		local_min = glm::min(local_min, v.position);
		local_max = glm::max(local_max, v.position);
	}

	upload_display();
}
